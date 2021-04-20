


class LoggingMixin():
    """
    Mixin class with methods for logging of configurations.
    """
    
    def get_option_and_log(self, key):
        
        value = self.get_option(key)
        self.config.log(f"{self.configuration_key}.{key} set to {value}")
        
        return value
