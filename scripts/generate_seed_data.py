import csv
from collections import Counter
from pathlib import Path


OUTPUT = Path("data/generated_router_train.csv")
FIELDNAMES = ["question", "workflow", "tools"]

# Data design notes:
# - workflow means execution flow, not a detailed life topic.
# - Casual campus-life requests such as "我饿了", "想吃饭了", "我想打球", and
#   "我不舒服" are still campus knowledge questions, so they are labeled
#   KNOWLEDGE_QA.
# - Time needs are represented by tools. For example:
#   "我饿了" -> KNOWLEDGE_QA, SEARCH_PUBLIC_KNOWLEDGE
#   "我现在饿了" -> KNOWLEDGE_QA, GET_CURRENT_TIME|SEARCH_PUBLIC_KNOWLEDGE
#   "明天食堂开门吗" -> KNOWLEDGE_QA, GET_CURRENT_TIME|SEARCH_PUBLIC_KNOWLEDGE
# - Personal planning questions such as "明天有什么作业要交吗" and
#   "我下周什么时候有空" are AGENT_TASK because they need personal schedule,
#   todo, time, and planning context.

PUBLIC_KNOWLEDGE = "SEARCH_PUBLIC_KNOWLEDGE"
TIME_PUBLIC_KNOWLEDGE = "GET_CURRENT_TIME|SEARCH_PUBLIC_KNOWLEDGE"
AGENT_TIME_TOOLS = "GET_CURRENT_TIME|SEARCH_USER_SCHEDULE|SEARCH_USER_TODO|SEARCH_GLOBAL_KNOWLEDGE"

SEED_ROWS = {
    "KNOWLEDGE_QA": [
        ("我饿了", PUBLIC_KNOWLEDGE),
        ("想吃饭了", PUBLIC_KNOWLEDGE),
        ("学校哪里能吃饭", PUBLIC_KNOWLEDGE),
        ("附近有什么吃的", PUBLIC_KNOWLEDGE),
        ("我想去饭堂", PUBLIC_KNOWLEDGE),
        ("哪里有饭吃", PUBLIC_KNOWLEDGE),
        ("我想打球", PUBLIC_KNOWLEDGE),
        ("想运动一下", PUBLIC_KNOWLEDGE),
        ("我不舒服", PUBLIC_KNOWLEDGE),
        ("我想去校医院", PUBLIC_KNOWLEDGE),
        ("快递到了去哪拿", PUBLIC_KNOWLEDGE),
        ("校园卡没钱了", PUBLIC_KNOWLEDGE),
        ("我想洗澡", PUBLIC_KNOWLEDGE),
        ("宿舍没电了怎么办", PUBLIC_KNOWLEDGE),
        ("华工有哪些食堂", PUBLIC_KNOWLEDGE),
        ("校园卡丢了怎么办", PUBLIC_KNOWLEDGE),
        ("图书馆在哪里", PUBLIC_KNOWLEDGE),
        ("校医院几点上班", PUBLIC_KNOWLEDGE),
        ("学生证丢了怎么补办", PUBLIC_KNOWLEDGE),
        ("宿舍用电怎么充值", PUBLIC_KNOWLEDGE),
        ("明天食堂开门吗", TIME_PUBLIC_KNOWLEDGE),
        ("今天图书馆开放吗", TIME_PUBLIC_KNOWLEDGE),
        ("周末体育馆能去吗", TIME_PUBLIC_KNOWLEDGE),
        ("晚上还能去饭堂吗", TIME_PUBLIC_KNOWLEDGE),
        ("下周一教务处上班吗", TIME_PUBLIC_KNOWLEDGE),
        ("国庆校医院开不开", TIME_PUBLIC_KNOWLEDGE),
        ("我现在饿了", TIME_PUBLIC_KNOWLEDGE),
        ("今晚图书馆几点关门", TIME_PUBLIC_KNOWLEDGE),
    ],
    "AGENT_TASK": [
        ("明天有什么作业要交吗", AGENT_TIME_TOOLS),
        ("我明天有哪些事情", AGENT_TIME_TOOLS),
        ("我下周什么时候有空", AGENT_TIME_TOOLS),
        ("帮我安排明天的学习", AGENT_TIME_TOOLS),
        ("帮我看看今天怎么规划", AGENT_TIME_TOOLS),
        ("根据我的课程表安排明天", AGENT_TIME_TOOLS),
        ("帮我制定期末复习计划", AGENT_TIME_TOOLS),
        ("今晚先写什么作业", AGENT_TIME_TOOLS),
        ("帮我制定明天的日程表", AGENT_TIME_TOOLS),
        ("明天我应该先做什么", AGENT_TIME_TOOLS),
        ("帮我整理今天的待办事项", AGENT_TIME_TOOLS),
        ("安排这周的复习时间", AGENT_TIME_TOOLS),
        ("把下周的作业和考试排一个优先级", AGENT_TIME_TOOLS),
        ("生成一份数据库课程复习计划", "SEARCH_CONVERSATION_KNOWLEDGE|SEARCH_GLOBAL_KNOWLEDGE"),
        ("根据我的空闲时间安排健身和学习", "SEARCH_USER_SCHEDULE|SEARCH_USER_TODO"),
        ("帮我拆解这个项目的执行步骤", "SEARCH_CONVERSATION_KNOWLEDGE|SEARCH_GLOBAL_KNOWLEDGE"),
        ("把明天的会议和学习任务排进日程", AGENT_TIME_TOOLS),
        ("为我制定一份英语六级备考计划", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("安排今晚两小时的高效复习流程", "GET_CURRENT_TIME|SEARCH_USER_TODO|SEARCH_GLOBAL_KNOWLEDGE"),
        ("根据截止日期整理任务清单", "SEARCH_USER_TODO|SEARCH_GLOBAL_KNOWLEDGE"),
        ("帮我判断明天下午有没有空", "GET_CURRENT_TIME|SEARCH_USER_SCHEDULE|SEARCH_USER_TODO"),
        ("规划一下这周每天复习什么", AGENT_TIME_TOOLS),
        ("按优先级整理我的任务", "SEARCH_USER_TODO|SEARCH_GLOBAL_KNOWLEDGE"),
        ("帮我安排考试周的学习节奏", "SEARCH_USER_SCHEDULE|SEARCH_USER_TODO|SEARCH_GLOBAL_KNOWLEDGE"),
        ("看看今天还有哪些任务没完成", "GET_CURRENT_TIME|SEARCH_USER_TODO"),
    ],
    "FILE_QA": [
        ("总结我刚刚上传的文件", "SEARCH_FILE_CONTENT"),
        ("这个文档主要讲了什么", "SEARCH_FILE_CONTENT"),
        ("我上传的课程表里周三有什么课", "SEARCH_FILE_CONTENT"),
        ("根据上传文件回答问题", "SEARCH_FILE_CONTENT"),
        ("文件里有没有提到考试", "SEARCH_FILE_CONTENT"),
        ("帮我提取上传文件中的重点", "SEARCH_FILE_CONTENT"),
        ("概括这份课程资料的重点", "SEARCH_FILE_CONTENT"),
        ("从上传文件中提取关键知识点", "SEARCH_FILE_CONTENT"),
        ("分析一下我上传的实验报告", "SEARCH_FILE_CONTENT"),
        ("根据刚才上传的课程表回答", "SEARCH_FILE_CONTENT"),
        ("这份 PDF 里有哪些重要结论", "SEARCH_FILE_CONTENT"),
        ("整理上传文档中的待办事项", "SEARCH_FILE_CONTENT"),
        ("从这份附件中找出时间和地点", "SEARCH_FILE_CONTENT"),
        ("根据上传的论文说明研究方法", "SEARCH_FILE_CONTENT"),
        ("检查这个文件里是否包含报名要求", "SEARCH_FILE_CONTENT"),
        ("把我上传的会议记录整理成摘要", "SEARCH_FILE_CONTENT"),
        ("这个文件中老师布置了什么作业", "SEARCH_FILE_CONTENT"),
        ("刚上传的图片里有什么信息", "SEARCH_FILE_CONTENT"),
        ("帮我看一下这份表格", "SEARCH_FILE_CONTENT"),
        ("提取附件里的截止日期", "SEARCH_FILE_CONTENT"),
        ("根据这份材料生成复习提纲", "SEARCH_FILE_CONTENT"),
        ("这个课件讲了哪些概念", "SEARCH_FILE_CONTENT"),
        ("上传文件里有没有提到地点", "SEARCH_FILE_CONTENT"),
        ("这份资料适合怎么复习", "SEARCH_FILE_CONTENT"),
        ("帮我检查文档中的报名流程", "SEARCH_FILE_CONTENT"),
    ],
    "USER_KNOWLEDGE_QA": [
        ("查询我的全局知识库", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("看看我之前上传过什么资料", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("从我的个人知识库里找操作系统资料", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("我保存过哪些课程表", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("查询我之前导入的文件内容", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("从我的知识库里找复习内容", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("查找我之前保存的待办事项", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("看看我的全局资料里有没有考试安排", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("搜索我的个人数据库", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("查询我上传过的课程资料", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("查一下我保存的项目笔记", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("从我的历史资料里找报名信息", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("检索我的个人文档中关于实习的内容", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("看看我之前记录的会议纪要", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("在我的长期知识库里找论文选题", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("查询我的历史导入资料", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("我以前保存过校医院信息吗", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("从我的资料里找数据库笔记", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("查看我的个人知识库课程安排", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("帮我搜一下以前导入的复习资料", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("我之前收藏过哪些回答", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("找找我的知识库里有没有社团资料", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("查询我的长期保存内容", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("从我的全局知识里找课程表", "SEARCH_GLOBAL_KNOWLEDGE"),
        ("看看我保存的考试重点", "SEARCH_GLOBAL_KNOWLEDGE"),
    ],
    "DIRECT_AI": [
        ("给我讲个笑话", "DIRECT_LLM"),
        ("解释一下人工智能", "DIRECT_LLM"),
        ("推荐几本适合大学生看的书", "DIRECT_LLM"),
        ("帮我想一个项目名字", "DIRECT_LLM"),
        ("什么是操作系统", "DIRECT_LLM"),
        ("解释一下数据库", "DIRECT_LLM"),
        ("机器学习和深度学习有什么区别", "DIRECT_LLM"),
        ("数据库索引有什么用", "DIRECT_LLM"),
        ("如何提高英语口语能力", "DIRECT_LLM"),
        ("写一句鼓励自己的话", "DIRECT_LLM"),
        ("解释一下什么是时间复杂度", "DIRECT_LLM"),
        ("给我一个学习编程的建议", "DIRECT_LLM"),
        ("说明一下碳中和的含义", "DIRECT_LLM"),
        ("介绍一下区块链的基本概念", "DIRECT_LLM"),
        ("如何培养长期阅读习惯", "DIRECT_LLM"),
        ("写一段面试自我介绍", "DIRECT_LLM"),
        ("帮我起一个社团活动名字", "DIRECT_LLM"),
        ("解释一下什么是编译原理", "DIRECT_LLM"),
        ("推荐几部放松的电影", "DIRECT_LLM"),
        ("给我一个晨读计划建议", "DIRECT_LLM"),
        ("写一段活动开场白", "DIRECT_LLM"),
        ("什么是神经网络", "DIRECT_LLM"),
        ("解释一下缓存的作用", "DIRECT_LLM"),
        ("帮我想一句海报标语", "DIRECT_LLM"),
        ("讲讲怎样提升表达能力", "DIRECT_LLM"),
    ],
    "SYSTEM_ACTION": [
        ("删除这个对话", "WRITE_USER_DATA"),
        ("修改我的昵称", "WRITE_USER_DATA"),
        ("把这个文件加入全局知识库", "WRITE_USER_DATA"),
        ("删除这条知识", "WRITE_USER_DATA"),
        ("把当前对话知识转为全局知识", "WRITE_USER_DATA"),
        ("修改界面为黑夜模式", "WRITE_USER_DATA"),
        ("保存我的个人资料", "WRITE_USER_DATA"),
        ("把这份资料保存到我的知识库", "WRITE_USER_DATA"),
        ("删除我刚刚上传的文件", "WRITE_USER_DATA"),
        ("清空当前会话记录", "WRITE_USER_DATA"),
        ("把我的学院改成计算机学院", "WRITE_USER_DATA"),
        ("更新我的年级信息", "WRITE_USER_DATA"),
        ("取消这条待办事项", "WRITE_USER_DATA"),
        ("把这个回答收藏起来", "WRITE_USER_DATA"),
        ("移除这份全局知识", "WRITE_USER_DATA"),
        ("保存当前聊天内容", "WRITE_USER_DATA"),
        ("把我的手机号改一下", "WRITE_USER_DATA"),
        ("删除这份课程资料", "WRITE_USER_DATA"),
        ("把这个文件移出知识库", "WRITE_USER_DATA"),
        ("保存我的偏好设置", "WRITE_USER_DATA"),
        ("修改我的个人简介", "WRITE_USER_DATA"),
        ("把当前文件设为公开知识", "WRITE_USER_DATA"),
        ("撤销刚才的保存", "WRITE_USER_DATA"),
        ("把这条记录删掉", "WRITE_USER_DATA"),
        ("更新我的班级信息", "WRITE_USER_DATA"),
    ],
    "UNKNOWN": [
        ("这个怎么办", "SEARCH_CONVERSATION_HISTORY"),
        ("那个还能用吗", "SEARCH_CONVERSATION_HISTORY"),
        ("你帮我看一下", "SEARCH_CONVERSATION_HISTORY"),
        ("这个呢", "SEARCH_CONVERSATION_HISTORY"),
        ("继续", "SEARCH_CONVERSATION_HISTORY"),
        ("上面那个", "SEARCH_CONVERSATION_HISTORY"),
        ("刚才说的可以吗", "SEARCH_CONVERSATION_HISTORY"),
        ("这里要怎么处理", "SEARCH_CONVERSATION_HISTORY"),
        ("这样行不行", "SEARCH_CONVERSATION_HISTORY"),
        ("帮我弄一下", "SEARCH_CONVERSATION_HISTORY"),
        ("还是之前那个", "SEARCH_CONVERSATION_HISTORY"),
        ("接着来", "SEARCH_CONVERSATION_HISTORY"),
        ("那一步怎么做", "SEARCH_CONVERSATION_HISTORY"),
        ("再试一次", "SEARCH_CONVERSATION_HISTORY"),
        ("把它改一下", "SEARCH_CONVERSATION_HISTORY"),
        ("我说的是那个", "SEARCH_CONVERSATION_HISTORY"),
        ("这是什么意思", "SEARCH_CONVERSATION_HISTORY"),
        ("按刚才那个来", "SEARCH_CONVERSATION_HISTORY"),
        ("你觉得呢", "SEARCH_CONVERSATION_HISTORY"),
        ("可以这样吗", "SEARCH_CONVERSATION_HISTORY"),
        ("帮我继续处理", "SEARCH_CONVERSATION_HISTORY"),
        ("刚刚那个再看下", "SEARCH_CONVERSATION_HISTORY"),
        ("这块有问题吗", "SEARCH_CONVERSATION_HISTORY"),
        ("上面的内容怎么办", "SEARCH_CONVERSATION_HISTORY"),
        ("然后呢", "SEARCH_CONVERSATION_HISTORY"),
    ],
}

INFO_PREFIXES = ["", "请问", "我想知道", "麻烦告诉我", "能不能告诉我"]
HELP_PREFIXES = ["帮我", "请帮我", "麻烦帮我", "能不能帮我", "可以帮我"]
ACTION_PREFIXES = ["", "请帮我", "麻烦", "帮我", "请"]
UNKNOWN_PREFIXES = ["", "请问", "我想确认一下", "帮我看看", "麻烦看一下"]

SPECIAL_EXPANSIONS = {
    "我饿了": ["我饿了", "我有点饿了", "我饿了去哪吃饭", "我饿了学校哪里有吃的", "现在有点饿去哪吃"],
    "想吃饭了": ["想吃饭了", "现在想吃饭了", "想吃饭去哪比较方便", "学校哪里适合吃饭", "附近哪里能吃饭"],
    "学校哪里能吃饭": ["学校哪里能吃饭", "学校哪里可以吃饭", "校内哪里能吃饭", "校园里哪里有吃的", "在学校去哪吃饭"],
    "附近有什么吃的": ["附近有什么吃的", "学校附近有什么吃的", "附近哪里有饭吃", "周边有什么能吃的", "附近吃饭去哪"],
    "我想去饭堂": ["我想去饭堂", "想去饭堂吃饭", "饭堂在哪里", "学校饭堂怎么去", "现在想去饭堂"],
    "哪里有饭吃": ["哪里有饭吃", "学校哪里有饭吃", "校内哪里能吃饭", "附近哪里有饭吃", "饭点去哪吃饭"],
    "我想打球": ["我想打球", "学校哪里能打球", "想找地方打球", "校内哪里可以打球", "我想去运动场打球"],
    "想运动一下": ["想运动一下", "学校哪里能运动", "想找地方运动一下", "校内哪里适合运动", "想去体育馆运动"],
    "我不舒服": ["我不舒服", "身体不舒服去哪看", "不舒服可以去哪里", "我不舒服想看医生", "学校哪里能看病"],
    "我想去校医院": ["我想去校医院", "校医院在哪里", "学校医院怎么去", "我想去校医院看一下", "校医院怎么走"],
    "快递到了去哪拿": ["快递到了去哪拿", "学校快递去哪取", "快递点在哪里", "取快递去哪里", "我的快递到了在哪里拿"],
    "校园卡没钱了": ["校园卡没钱了", "校园卡怎么充值", "校园卡余额不够怎么办", "学校校园卡在哪里充钱", "校园卡没钱去哪里处理"],
    "我想洗澡": ["我想洗澡", "学校哪里能洗澡", "宿舍洗澡怎么弄", "洗澡热水怎么用", "校内洗澡去哪里"],
    "宿舍没电了怎么办": ["宿舍没电了怎么办", "宿舍断电了怎么办", "宿舍电费怎么充", "宿舍没电去哪处理", "宿舍用电不够了怎么办"],
}


def expand_question(question: str) -> list[str]:
    if question in SPECIAL_EXPANSIONS:
        return SPECIAL_EXPANSIONS[question]

    if question.startswith("帮我"):
        tail = question.removeprefix("帮我")
        return [f"{prefix}{tail}" for prefix in HELP_PREFIXES]

    if question.startswith("根据"):
        return [question, f"请{question}", f"麻烦{question}", f"帮我{question}", f"能不能{question}"]

    if question.startswith(("把", "从", "删除", "修改", "保存", "清空", "取消", "移除", "更新", "安排", "生成", "为我", "撤销")):
        return [f"{prefix}{question}" if prefix else question for prefix in ACTION_PREFIXES]

    if question in {"继续", "上面那个", "这个呢", "然后呢"} or "那个" in question:
        return [f"{prefix}{question}" if prefix else question for prefix in UNKNOWN_PREFIXES]

    return [f"{prefix}{question}" if prefix else question for prefix in INFO_PREFIXES]


def generate_rows() -> list[dict[str, str]]:
    rows = []
    seen_questions = set()

    for workflow, seed_rows in SEED_ROWS.items():
        for question, tools in seed_rows:
            for expanded_question in expand_question(question):
                if expanded_question in seen_questions:
                    continue
                seen_questions.add(expanded_question)
                rows.append({
                    "question": expanded_question,
                    "workflow": workflow,
                    "tools": tools,
                })

    return rows


def generate() -> None:
    rows = generate_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp_output = OUTPUT.with_suffix(".tmp")

    with open(temp_output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    temp_output.replace(OUTPUT)

    counts = Counter(row["workflow"] for row in rows)
    print("生成完成")
    print("总条数:", len(rows))
    for workflow in SEED_ROWS:
        print(f"{workflow}: {counts[workflow]}")
    print("输出文件:", OUTPUT)


if __name__ == "__main__":
    generate()
