class Node:
    def __init__(self, completed_word):
        self.letters = {}
        self.completed_word = completed_word


def add_word(node, word):
    prev_node = node
    for letter in word:
        if letter not in node.letters:
            node.letters[letter] = Node(False)
        prev_node = node
        node = node.letters[letter]
    prev_node.completed_word = True


def find_word(node, word):
    prev_node = node
    for letter in word:
        if letter not in node.letters:
            return False
        prev_node = node
        node = node.letters[letter]
    return prev_node.completed_word


if __name__ == '__main__':
    trie = Node(False)
    add_word(trie, "hello")
    print(find_word(trie, "hello"))
