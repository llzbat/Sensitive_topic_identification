import csv

BLACK_WORD_PATH = 'Sensitive_Word/black_words.csv'  # 修改为CSV文件路径
STOP_WORD_PATH = 'Sensitive_Word/stop_words.txt'
T2S_PATH = 'Sensitive_Word/t2s.txt'


class DFAFilter(object):
    """基于DFA算法的敏感词过滤系统"""
    def __init__(self):
        super(DFAFilter, self).__init__()
        self.black_words = self.read_black_words_file(BLACK_WORD_PATH)
        self.stop_words = set(self.read_list_file(STOP_WORD_PATH))  # 使用集合提高查找效率
        self.black_word_chains = {}
        self.delimit = '\x00'
        self.parse_sensitive_words()

    def read_list_file(self, path):
        """从文件中读取停顿词列表"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = (line.strip() for line in f if line.strip())  # 使用生成器表达式
                return set(lines)  # 返回生成器表达式的结果作为集合
        except IOError as e:
            print(f"Error reading file {path}: {e}")
            return set()

    def read_black_words_file(self, path):
        """从CSV文件中读取敏感词及其类别"""
        black_words = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        black_word, category = row
                        black_word = black_word.strip()
                        category = category.strip()
                        if black_word:
                            black_words[black_word] = category
        except IOError as e:
            print(f"Error reading file {path}: {e}")
        return black_words

    def parse_sensitive_words(self):
        """解析敏感词并构建敏感词链表"""
        for black_word, category in self.black_words.items():
            self.add_sensitive_words(black_word, category)

    def add_sensitive_words(self, black_word, category):
        """将敏感词及其类别添加到敏感词链表"""
        level = self.black_word_chains
        for char in black_word:
            if char not in level:
                level[char] = {}
            level = level[char]
        level[self.delimit] = category

    def filter_sensitive_words(self, content, replace="*", t2s=False):
        """对指定文本进行过滤"""
        filtered_content = []
        black_words = []
        idx = 0
        while idx < len(content):
            level = self.black_word_chains
            step_ins = 0
            message_chars = content[idx:]
            black_word = ''
            black_word_category = ''
            for char in message_chars:
                if char in self.stop_words and step_ins != 0:
                    step_ins += 1
                    continue
                if char in level:
                    step_ins += 1
                    black_word += char
                    if self.delimit in level[char]:
                        black_word_category = level[char][self.delimit]
                        black_words.append((black_word, black_word_category))
                        filtered_content.append(replace * step_ins)
                        idx += step_ins
                        break
                    else:
                        level = level[char]
                else:
                    filtered_content.append(content[idx])
                    break
            else:
                # 如果循环正常结束，则意味着没有找到敏感词
                filtered_content.append(content[idx])
            idx += 1
        return ''.join(filtered_content), black_words

