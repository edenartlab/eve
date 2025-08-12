import modal
from bson import ObjectId
from typing import Dict, Any, Optional, List, Tuple
import logging
import asyncio
from datetime import datetime, timezone


# Safe modal.Dict operations with retry logic (moved here to avoid circular imports)
async def safe_modal_get(modal_dict, key: str, default=None, retries: int = 3):
    """Safely get value from modal.Dict with exponential backoff retry."""
    for attempt in range(retries):
        try:
            return modal_dict[key]
        except KeyError:
            return default
        except Exception as e:
            if attempt == retries - 1:
                logging.warning(f"Modal.Dict get failed after {retries} attempts for key {key}: {e}")
                return default
            await asyncio.sleep(0.1 * (2 ** attempt))
    return default

async def safe_modal_set(modal_dict, key: str, value, retries: int = 3):
    """Safely set value in modal.Dict with exponential backoff retry."""
    for attempt in range(retries):
        try:
            modal_dict[key] = value
            return True
        except Exception as e:
            if attempt == retries - 1:
                logging.warning(f"Modal.Dict set failed after {retries} attempts for key {key}: {e}")
                return False
            await asyncio.sleep(0.1 * (2 ** attempt))
    return False

async def safe_modal_batch_get(modal_dicts_and_keys: list, default=None) -> list:
    """Batch get from multiple modal.Dict objects with retry logic.
    
    Args:
        modal_dicts_and_keys: List of tuples (modal_dict, key, default_value)
        
    Returns:
        List of values in same order as input
    """
    async def get_single(modal_dict, key, default_val):
        return await safe_modal_get(modal_dict, key, default_val)
    
    tasks = [get_single(modal_dict, key, default_val) 
             for modal_dict, key, default_val in modal_dicts_and_keys]
    
    return await asyncio.gather(*tasks)


class ModalDictState:
    """
    A wrapper class for managing modal.Dict state with consistent patterns for:
    - Fetching the latest version of modal dicts
    - Creating local copies for function context
    - Indexing with keys (session_id, agent_id, user_id)
    - Updating values
    - Batch operations for performance
    """
    
    def __init__(self, modal_dict: modal.Dict, name: str):
        self.modal_dict = modal_dict
        self.name = name
    
    async def get_agent_dict(self, agent_id: ObjectId, default: Dict = None) -> Dict[str, Any]:
        """Get agent dictionary, creating default if missing."""
        if default is None:
            default = {}
        
        agent_key = str(agent_id)
        agent_dict = await safe_modal_get(self.modal_dict, agent_key, default)
        
        if not agent_dict:
            print(f"No {self.name} dict found for agent {agent_key}, creating empty dict")
            await safe_modal_set(self.modal_dict, agent_key, default)
            return default
        
        return agent_dict
    
    async def update_agent_dict(self, agent_id: ObjectId, updated_dict: Dict[str, Any]) -> bool:
        """Update entire agent dictionary."""
        agent_key = str(agent_id)
        return await safe_modal_set(self.modal_dict, agent_key, updated_dict)
    
    async def get_session_value(self, agent_id: ObjectId, session_id: ObjectId, key: str, default: Any = None) -> Any:
        """Get a specific value from a session within an agent's dict."""
        agent_dict = await self.get_agent_dict(agent_id)
        session_key = str(session_id)
        session_dict = agent_dict.get(session_key, {})
        return session_dict.get(key, default)
    
    async def update_session_value(self, agent_id: ObjectId, session_id: ObjectId, key: str, value: Any) -> bool:
        """Update a specific value in a session within an agent's dict."""
        agent_dict = await self.get_agent_dict(agent_id)
        session_key = str(session_id)
        
        if session_key not in agent_dict:
            agent_dict[session_key] = {}
        
        agent_dict[session_key][key] = value
        return await self.update_agent_dict(agent_id, agent_dict)
    
    async def get_user_value(self, agent_id: ObjectId, user_id: ObjectId, key: str, default: Any = None) -> Any:
        """Get a specific value for a user within an agent's dict."""
        agent_dict = await self.get_agent_dict(agent_id)
        user_key = str(user_id)
        user_dict = agent_dict.get(user_key, {})
        return user_dict.get(key, default)
    
    async def update_user_value(self, agent_id: ObjectId, user_id: ObjectId, key: str, value: Any) -> bool:
        """Update a specific value for a user within an agent's dict."""
        agent_dict = await self.get_agent_dict(agent_id)
        user_key = str(user_id)
        
        if user_key not in agent_dict:
            agent_dict[user_key] = {}
        
        agent_dict[user_key][key] = value
        return await self.update_agent_dict(agent_id, agent_dict)
    
    async def get_agent_value(self, agent_id: ObjectId, key: str, default: Any = None) -> Any:
        """Get a specific value from an agent's dict."""
        agent_dict = await self.get_agent_dict(agent_id)
        return agent_dict.get(key, default)
    
    async def update_agent_value(self, agent_id: ObjectId, key: str, value: Any) -> bool:
        """Update a specific value in an agent's dict."""
        agent_dict = await self.get_agent_dict(agent_id)
        agent_dict[key] = value
        return await self.update_agent_dict(agent_id, agent_dict)
    
    async def batch_get_agent_dicts(self, agent_ids: List[ObjectId]) -> List[Dict[str, Any]]:
        """Batch get multiple agent dictionaries."""
        agent_keys = [str(agent_id) for agent_id in agent_ids]
        batch_requests = [(self.modal_dict, key, {}) for key in agent_keys]
        return await safe_modal_batch_get(batch_requests)
    
    async def batch_update_agent_dicts(self, updates: List[Tuple[ObjectId, Dict[str, Any]]]) -> List[bool]:
        """Batch update multiple agent dictionaries."""
        tasks = []
        for agent_id, agent_dict in updates:
            tasks.append(self.update_agent_dict(agent_id, agent_dict))
        return await asyncio.gather(*tasks)


class MultiModalDictState:
    """
    A wrapper for managing multiple modal.Dict objects with batch operations.
    """
    
    def __init__(self, modal_dicts: Dict[str, ModalDictState]):
        self.modal_dicts = modal_dicts
    
    def get_dict(self, name: str) -> ModalDictState:
        """Get a specific modal dict by name."""
        return self.modal_dicts[name]
    
    async def batch_get_from_multiple_dicts(self, requests: List[Tuple[str, ObjectId, Optional[str], Any]]) -> List[Any]:
        """
        Batch get from multiple modal dicts.
        
        Args:
            requests: List of tuples (dict_name, agent_id, optional_key, default_value)
                     If optional_key is None, gets entire agent dict
                     If optional_key is provided, gets specific value from agent dict
        
        Returns:
            List of values in same order as requests
        """
        batch_requests = []
        
        for dict_name, agent_id, optional_key, default_value in requests:
            modal_dict = self.modal_dicts[dict_name].modal_dict
            agent_key = str(agent_id)
            batch_requests.append((modal_dict, agent_key, default_value if optional_key is None else {}))
        
        results = await safe_modal_batch_get(batch_requests)
        
        # If optional_key was specified, extract that specific value
        final_results = []
        for i, (dict_name, agent_id, optional_key, default_value) in enumerate(requests):
            if optional_key is None:
                final_results.append(results[i])
            else:
                agent_dict = results[i]
                final_results.append(agent_dict.get(optional_key, default_value))
        
        return final_results