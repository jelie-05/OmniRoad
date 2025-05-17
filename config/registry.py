# src/config/registry.py
from typing import Dict, Type, Optional, TypeVar, Generic, Callable

T = TypeVar('T')  # Generic type for the configuration classes

class ConfigRegistry(Generic[T]):
    """Registry for configuration classes."""
    
    def __init__(self, name: str):
        """
        Initialize a new registry.
        
        Args:
            name: Name of this registry for error messages
        """
        self._registry: Dict[str, Type[T]] = {}
        self.name = name
    
    def register(self, name: Optional[str] = None) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a configuration class."""
        def decorator(config_cls: Type[T]) -> Type[T]:
            register_name = name or config_cls.__name__
            self._registry[register_name] = config_cls
            return config_cls
        return decorator
    
    def get(self, name: str) -> Type[T]:
        """Get a configuration class by name."""
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise KeyError(
                f"Configuration '{name}' not found in {self.name}. Available: {available}"
            )
        return self._registry[name]
    
    def list_available(self) -> Dict[str, str]:
        """List all available configurations with their descriptions."""
        return {
            name: self._get_description(config_cls)
            for name, config_cls in self._registry.items()
        }
    
    @staticmethod
    def _get_description(cls: Type[T]) -> str:
        """Extract the class description from its docstring."""
        if cls.__doc__:
            return cls.__doc__.split("\n")[0].strip()
        return str(cls)