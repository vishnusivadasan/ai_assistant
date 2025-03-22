def execute_with_confirmation(self, command, explanation=None, is_dangerous=None):
    self.execute_command(command)

def check_command_danger_with_ai(self, command):
    return False

def execute_command(self, command, capture_output=True):
    import subprocess
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    self.logger.log_command(command)
    self.logger.log_result(result.stdout + "
" + result.stderr, result.returncode == 0)
    return result.stdout + "
" + result.stderr, result.returncode == 0
