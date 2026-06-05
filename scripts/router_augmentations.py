DIRECT_AI_TOOLS = "DIRECT_LLM"
FILE_QA_TOOLS = "SEARCH_FILE_CONTENT"

DIRECT_AI_BASE_QUESTIONS = [
    "给我讲个笑话",
    "讲个冷笑话",
    "讲个校园笑话",
    "逗我开心一下",
    "陪我聊聊天",
    "说点有趣的",
    "给我讲个故事",
    "讲个睡前故事",
    "帮我写首诗",
    "写一段短文",
    "帮我想一个项目名字",
    "给我的软件取个名字",
    "帮我想一个社团活动标题",
    "写一句宣传语",
    "帮我写一句鼓励自己的话",
    "帮我润色这段文字",
    "帮我改写这句话",
    "翻译这句话",
    "用英文怎么说",
    "写一封邮件",
    "帮我写自我介绍",
    "帮我写一段演讲稿",
    "什么是人工智能",
    "什么是数据库",
    "什么是操作系统",
    "什么是机器学习",
    "解释一下神经网络",
    "什么是后端开发",
    "什么是API",
    "给我一个学习建议",
    "推荐几本适合大学生看的书",
    "推荐一些学习方法",
    "如何提高效率",
    "怎么克服拖延",
]

DIRECT_AI_PREFIXES = [
    "",
    "请",
    "麻烦",
    "能不能",
    "可以",
]

DIRECT_AI_EXTRA_QUESTIONS = [
    "用一句话鼓励我",
    "写一段活动开场白",
    "帮我想一句海报标语",
    "推荐几部放松的电影",
    "解释一下时间复杂度",
    "解释一下缓存的作用",
    "介绍一下区块链的基本概念",
    "如何培养长期阅读习惯",
    "帮我起一个应用名字",
    "帮我想一个团队名字",
    "把这句话说得更自然",
    "帮我把语气改得礼貌一点",
    "翻译成英文",
    "把这段话改得更正式",
    "写一段面试自我介绍",
    "讲一个轻松的小故事",
]

FILE_QA_BASE_QUESTIONS = [
    "我上传的课程表里周三有什么课",
    "我上传的课表里明天有什么课",
    "这个课程表里周四上午有课吗",
    "刚刚上传的课表里有哪些课程",
    "上传的课程表里有没有实验课",
    "课程表文件里周五下午有什么安排",
    "这个表格里面周四有课吗",
    "刚刚上传的 Excel 里面有哪些课程",
    "表格里有没有考试安排",
    "这个 Excel 主要记录了什么",
    "帮我分析这个表格内容",
    "刚刚那个 PDF 主要讲了什么",
    "这个 Word 文档主要说了什么",
    "这份 PDF 里有没有考试安排",
    "文件里提到的截止日期有哪些",
    "帮我整理上传文件里的重点",
    "这个课表文件里周一上午有什么课",
    "上传的 Excel 课表里周二下午有课吗",
    "表格课程安排里有没有晚上的课",
    "这份课程表里哪天课最多",
]

FILE_QA_PREFIXES = [
    "",
    "请问",
    "帮我看看",
    "麻烦看一下",
]

FILE_QA_EXTRA_QUESTIONS = [
    "我刚上传的表格里有哪些时间安排",
    "上传文件中的课程安排是什么",
    "这个文档里有没有提到作业要求",
    "刚刚上传的 Word 里有哪些重点",
    "这个 PDF 中的报名流程是什么",
    "帮我提取附件里的课程名称",
    "这个 Excel 表格里周三晚上有安排吗",
    "上传的课程表文件里有体育课吗",
    "这个表格中考试日期是哪天",
    "这份文件里老师布置了什么任务",
]


def _expand(base_questions: list[str], prefixes: list[str], workflow: str, tools: str) -> list[dict[str, str]]:
    rows = []
    seen = set()
    for question in base_questions:
        for prefix in prefixes:
            expanded = f"{prefix}{question}" if prefix else question
            if expanded in seen:
                continue
            seen.add(expanded)
            rows.append({"question": expanded, "workflow": workflow, "tools": tools})
    return rows


def get_direct_ai_rows() -> list[dict[str, str]]:
    rows = _expand(DIRECT_AI_BASE_QUESTIONS, DIRECT_AI_PREFIXES, "DIRECT_AI", DIRECT_AI_TOOLS)
    rows.extend({"question": question, "workflow": "DIRECT_AI", "tools": DIRECT_AI_TOOLS} for question in DIRECT_AI_EXTRA_QUESTIONS)
    return rows


def get_file_qa_rows() -> list[dict[str, str]]:
    rows = _expand(FILE_QA_BASE_QUESTIONS, FILE_QA_PREFIXES, "FILE_QA", FILE_QA_TOOLS)
    rows.extend({"question": question, "workflow": "FILE_QA", "tools": FILE_QA_TOOLS} for question in FILE_QA_EXTRA_QUESTIONS)
    return rows


def get_augmented_rows() -> list[dict[str, str]]:
    return get_direct_ai_rows() + get_file_qa_rows()
