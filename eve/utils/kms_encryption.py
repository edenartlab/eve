import os
import json
import base64
from typing import Optional, Dict, Any
from loguru import logger

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available, KMS encryption disabled")


class KMSEncryption:
    """Handle KMS encryption/decryption for deployment secrets"""

    def __init__(self):
        self.enabled = self._check_enabled()
        if self.enabled:
            self.kms_client = boto3.client("kms", region_name=self._get_region())
            self.key_id = self._get_key_id()
        else:
            self.kms_client = None
            self.key_id = None

    def _check_enabled(self) -> bool:
        """Check if KMS encryption is enabled"""
        if not BOTO3_AVAILABLE:
            return False

        # Only enable if AWS_KMS_KEY_ID or AWS_KMS_KEY_ARN is set
        return bool(os.getenv("AWS_KMS_KEY_ID") or os.getenv("AWS_KMS_KEY_ARN"))

    def _get_region(self) -> str:
        """Get AWS region from environment or default"""
        return os.getenv("AWS_REGION", "us-east-1")

    def _get_key_id(self) -> str:
        """Get KMS key ID or ARN from environment"""
        key_id = os.getenv("AWS_KMS_KEY_ARN") or os.getenv("AWS_KMS_KEY_ID")
        if not key_id:
            raise ValueError("AWS_KMS_KEY_ARN or AWS_KMS_KEY_ID must be set")
        return key_id

    def encrypt_secrets(self, secrets_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypt a secrets dictionary using KMS.

        Args:
            secrets_dict: Dictionary containing deployment secrets

        Returns:
            Dictionary with encrypted data and metadata

        Format:
        {
            "encrypted_data": base64-encoded encrypted JSON,
            "encryption_metadata": {
                "algorithm": "AES-256-GCM",
                "key_id": "kms-key-id",
                "encrypted": true
            }
        }
        """
        if not self.enabled:
            logger.debug("KMS encryption not enabled, storing secrets unencrypted")
            return secrets_dict

        if not secrets_dict:
            return secrets_dict

        try:
            # Convert secrets to JSON string
            secrets_json = json.dumps(secrets_dict)
            secrets_bytes = secrets_json.encode("utf-8")

            # Encrypt using KMS (envelope encryption)
            response = self.kms_client.encrypt(
                KeyId=self.key_id, Plaintext=secrets_bytes
            )

            # Base64 encode the ciphertext for MongoDB storage
            encrypted_data = base64.b64encode(response["CiphertextBlob"]).decode(
                "utf-8"
            )

            return {
                "encrypted_data": encrypted_data,
                "encryption_metadata": {
                    "algorithm": "AES-256-GCM",
                    "key_id": self.key_id,
                    "encrypted": True,
                },
            }

        except ClientError as e:
            logger.error(f"KMS encryption failed: {e}")
            raise Exception(f"Failed to encrypt secrets: {e}")

    def decrypt_secrets(
        self, encrypted_dict: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Decrypt secrets encrypted with KMS.

        Args:
            encrypted_dict: Dictionary with encrypted data and metadata

        Returns:
            Decrypted secrets dictionary or None if decryption fails
        """
        if not encrypted_dict:
            return None

        # Check if this is actually encrypted data
        if not isinstance(encrypted_dict, dict):
            return encrypted_dict

        if "encryption_metadata" not in encrypted_dict:
            # Not encrypted, return as-is (backward compatibility)
            return encrypted_dict

        if not encrypted_dict.get("encryption_metadata", {}).get("encrypted"):
            # Marked as not encrypted
            return encrypted_dict

        if not self.enabled:
            logger.error("KMS encryption not enabled but encrypted data found")
            raise Exception("Cannot decrypt secrets: KMS not configured")

        try:
            # Decode base64 encrypted data
            encrypted_data = encrypted_dict.get("encrypted_data")
            if not encrypted_data:
                logger.error("No encrypted_data found in encrypted secrets")
                return None

            ciphertext_blob = base64.b64decode(encrypted_data)

            # Decrypt using KMS
            response = self.kms_client.decrypt(CiphertextBlob=ciphertext_blob)

            # Parse decrypted JSON
            decrypted_bytes = response["Plaintext"]
            decrypted_json = decrypted_bytes.decode("utf-8")
            secrets_dict = json.loads(decrypted_json)

            return secrets_dict

        except ClientError as e:
            logger.error(f"KMS decryption failed: {e}")
            raise Exception(f"Failed to decrypt secrets: {e}")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse decrypted secrets: {e}")
            raise Exception(f"Failed to parse decrypted secrets: {e}")


# Global instance
_kms_encryption = None


def get_kms_encryption() -> KMSEncryption:
    """Get or create global KMS encryption instance"""
    global _kms_encryption
    if _kms_encryption is None:
        _kms_encryption = KMSEncryption()
    return _kms_encryption


def encrypt_deployment_secrets(
    secrets_dict: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to encrypt deployment secrets.

    Args:
        secrets_dict: Dictionary containing deployment secrets

    Returns:
        Encrypted dictionary or None if input is None
    """
    if secrets_dict is None:
        return None

    kms = get_kms_encryption()
    return kms.encrypt_secrets(secrets_dict)


def decrypt_deployment_secrets(
    encrypted_dict: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to decrypt deployment secrets.

    Args:
        encrypted_dict: Dictionary with encrypted data

    Returns:
        Decrypted dictionary or None if input is None
    """
    if encrypted_dict is None:
        return None

    kms = get_kms_encryption()
    return kms.decrypt_secrets(encrypted_dict)
