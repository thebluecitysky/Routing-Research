import nltk
import random
nltk.download('punkt')

from nltk.parse.stanford import StanfordParser

# 设置Stanford Parser的路径
stanford_parser_dir = '/path/to/stanford-parser-full-2020-11-17/'
stanford_parser_jar = stanford_parser_dir + 'stanford-parser.jar'
stanford_parser_model = stanford_parser_dir + 'stanford-parser-4.2.0-models.jar'

parser = StanfordParser(path_to_jar=stanford_parser_jar, path_to_models_jar=stanford_parser_model)

def swap_siblings(tree):
    for subtree in tree:
        if isinstance(subtree, nltk.Tree):
            if len(subtree) > 1:
                # 交换前两个子节点
                subtree[0], subtree[1] = subtree[1], subtree[0]
            swap_siblings(subtree)
    return tree


def replace_subtree(tree, depth_threshold=2):
    if isinstance(tree, nltk.Tree):
        if tree.height() > depth_threshold:
            # 创建一个新的相同类型的子树
            new_tree = nltk.Tree(tree.label(), [])
            for child in tree:
                if isinstance(child, nltk.Tree):
                    new_tree.append(replace_subtree(child, depth_threshold))
                else:
                    new_tree.append(child)
            return new_tree
        else:
            return tree
    else:
        return tree


# 定义生成变异查询的函数
def generate_variant_query(q):
    """
    对查询 q 进行句法变异，生成新的查询 q'。
    """
    syntax_tree = parser.parse(q)
    
    # 随机选择一种变异方法
    if random.choice([True, False]):
        modified_tree = swap_siblings(syntax_tree)
    else:
        modified_tree = replace_subtree(syntax_tree)
    
    # 将修改后的句法树转换回文本
    q_prime = parser.tree_to_text(modified_tree)
    return q_prime