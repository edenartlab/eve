import modal
from bson import ObjectId
from typing import Dict, Any, Optional, List, Tuple
import logging
import asyncio

# Safe modal.Dict operations with retry logic
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

async def safe_modal_batch_get(modal_dicts_and_keys: list) -> list:
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
    A wrapper class for managing modal.Dict state with generic patterns for:
    - Fetching the latest version of modal dicts
    - Creating local copies for function context
    - Indexing with arbitrary key paths
    - Updating values at any depth
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
    
    def _navigate_to_nested_value(self, data: Dict, key_path: List[str], default: Any = None) -> Any:
        """Navigate to a nested value using a list of keys."""
        current = data
        for key in key_path:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current
    
    def _set_nested_value(self, data: Dict, key_path: List[str], value: Any) -> None:
        """Set a nested value using a list of keys, creating intermediate dicts as needed."""
        current = data
        for key in key_path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[key_path[-1]] = value
    
    async def get_value(self, agent_id: ObjectId, key_path: List[str], default: Any = None) -> Any:
        """
        Get a value at any depth using a key path.
        
        Examples:
        - get_value(agent_id, ["session_123", "temperature"]) -> session value
        - get_value(agent_id, ["user_456", "preferences"]) -> user value  
        - get_value(agent_id, ["global_setting"]) -> agent-level value
        """
        agent_dict = await self.get_agent_dict(agent_id)
        return self._navigate_to_nested_value(agent_dict, key_path, default)
    
    async def update_value(self, agent_id: ObjectId, key_path: List[str], value: Any) -> bool:
        """
        Update a value at any depth using a key path.
        
        Examples:
        - update_value(agent_id, ["session_123", "temperature"], 0.8)
        - update_value(agent_id, ["user_456", "preferences"], {...})
        - update_value(agent_id, ["global_setting"], value)
        """
        agent_dict = await self.get_agent_dict(agent_id)
        self._set_nested_value(agent_dict, key_path, value)
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
    
    async def batch_get_values(self, requests: List[Tuple[ObjectId, List[str], Any]]) -> List[Any]:
        """
        Batch get multiple values using key paths.
        
        Args:
            requests: List of tuples (agent_id, key_path, default_value)
        
        Returns:
            List of values in same order as requests
        """
        agent_ids = [agent_id for agent_id, _, _ in requests]
        agent_dicts = await self.batch_get_agent_dicts(agent_ids)
        
        results = []
        for i, (_, key_path, default_value) in enumerate(requests):
            results.append(self._navigate_to_nested_value(agent_dicts[i], key_path, default_value))
        
        return results
    
    async def batch_update_values(self, updates: List[Tuple[ObjectId, List[str], Any]]) -> List[bool]:
        """
        Batch update multiple values using key paths.
        
        Args:
            updates: List of tuples (agent_id, key_path, value)
        
        Returns:
            List of success booleans in same order as updates
        """
        agent_ids = [agent_id for agent_id, _, _ in updates]
        agent_dicts = await self.batch_get_agent_dicts(agent_ids)
        
        # Apply updates to local copies
        for i, (_, key_path, value) in enumerate(updates):
            self._set_nested_value(agent_dicts[i], key_path, value)
        
        # Batch update all modified dicts
        dict_updates = [(agent_ids[i], agent_dicts[i]) for i in range(len(agent_ids))]
        return await self.batch_update_agent_dicts(dict_updates)


class MultiModalDictState:
    """
    A wrapper for managing multiple modal.Dict objects with batch operations.
    """
    
    def __init__(self, modal_dicts: Dict[str, ModalDictState]):
        self.modal_dicts = modal_dicts
    
    def get_dict(self, name: str) -> ModalDictState:
        """Get a specific modal dict by name."""
        return self.modal_dicts[name]
    
    async def batch_get_from_multiple_dicts(self, requests: List[Tuple[str, ObjectId, List[str], Any]]) -> List[Any]:
        """
        Batch get from multiple modal dicts using key paths.
        
        Args:
            requests: List of tuples (dict_name, agent_id, key_path, default_value)
                     key_path is a list of keys for nested access (empty list for entire agent dict)
        
        Returns:
            List of values in same order as requests
        """
        # Group requests by dict_name for efficiency
        dict_requests = {}
        for i, (dict_name, agent_id, key_path, default_value) in enumerate(requests):
            if dict_name not in dict_requests:
                dict_requests[dict_name] = []
            dict_requests[dict_name].append((i, agent_id, key_path, default_value))
        
        # Execute batch gets for each dict
        results = [None] * len(requests)
        
        for dict_name, dict_specific_requests in dict_requests.items():
            modal_dict_state = self.modal_dicts[dict_name]
            
            if all(len(key_path) == 0 for _, _, key_path, _ in dict_specific_requests):
                # All requests want entire agent dicts
                agent_ids = [agent_id for _, agent_id, _, _ in dict_specific_requests]
                agent_dicts = await modal_dict_state.batch_get_agent_dicts(agent_ids)
                for j, (original_index, _, _, _) in enumerate(dict_specific_requests):
                    results[original_index] = agent_dicts[j]
            else:
                # Mixed requests - use batch_get_values
                batch_requests = [(agent_id, key_path, default_value) 
                                for _, agent_id, key_path, default_value in dict_specific_requests]
                values = await modal_dict_state.batch_get_values(batch_requests)
                for j, (original_index, _, _, _) in enumerate(dict_specific_requests):
                    results[original_index] = values[j]
        
        return results
    
    async def batch_update_multiple_dicts(self, updates: List[Tuple[str, ObjectId, List[str], Any]]) -> List[bool]:
        """
        Batch update across multiple modal dicts using key paths.
        
        Args:
            updates: List of tuples (dict_name, agent_id, key_path, value)
        
        Returns:
            List of success booleans in same order as updates
        """
        # Group updates by dict_name for efficiency
        dict_updates = {}
        for i, (dict_name, agent_id, key_path, value) in enumerate(updates):
            if dict_name not in dict_updates:
                dict_updates[dict_name] = []
            dict_updates[dict_name].append((i, agent_id, key_path, value))
        
        # Execute batch updates for each dict
        results = [False] * len(updates)
        
        for dict_name, dict_specific_updates in dict_updates.items():
            modal_dict_state = self.modal_dicts[dict_name]
            batch_updates = [(agent_id, key_path, value) 
                           for _, agent_id, key_path, value in dict_specific_updates]
            update_results = await modal_dict_state.batch_update_values(batch_updates)
            
            for j, (original_index, _, _, _) in enumerate(dict_specific_updates):
                results[original_index] = update_results[j]
        
        return results


# Utility functions for creating key paths
def session_key_path(session_id: ObjectId, key: str) -> List[str]:
    """Create key path for session-scoped value."""
    return [str(session_id), key]

def user_key_path(user_id: ObjectId, key: str) -> List[str]:
    """Create key path for user-scoped value."""
    return [str(user_id), key]

def agent_key_path(key: str) -> List[str]:
    """Create key path for agent-scoped value."""
    return [key]

def nested_key_path(*keys) -> List[str]:
    """Create key path for arbitrary nested access."""
    return [str(key) for key in keys]