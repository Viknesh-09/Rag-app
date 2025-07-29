# === Chat Memory ===
class ChatMemory:
    def __init__(self):
        self.chat_history = []

    def add_message(self, user, message):
        self.chat_history.append({'user': user, 'message': message})

    def get_history(self):
        return self.chat_history
