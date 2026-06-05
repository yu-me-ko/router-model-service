from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutePostProcessResult:
    workflow: str
    corrected_by_rule: bool
    rule_name: str | None
    raw_workflow: str
    raw_confidence: float | None = None

    def __iter__(self):
        # Backward compatibility for older callers:
        # workflow, corrected = post_process_route(...)
        yield self.workflow
        yield self.corrected_by_rule


TIME_KEYWORDS = [
    "今天",
    "明天",
    "后天",
    "周末",
    "下周",
    "下个星期",
    "几点",
    "什么时候",
    "日期",
    "时间",
    "现在",
    "当前",
    "此刻",
    "马上",
    "今晚",
    "今天晚上",
    "晚上",
    "早上",
    "明早",
    "上午",
    "中午",
    "下午",
    "凌晨",
    "这个点",
    "这个时间",
    "前一周",
    "后一周",
    "来得及",
]

SCHEDULE_KEYWORDS = ["课表", "课程表", "课程安排", "有空", "空闲", "空档", "比较空", "哪段时间", "安排", "日程", "规划"]
TODO_KEYWORDS = ["待办", "作业", "截止", "任务", "先做", "先写", "提交"]

SYSTEM_ACTION_KEYWORDS = [
    "删除",
    "删掉",
    "移除",
    "不要留下",
    "不保留",
    "取消保存",
    "加入全局",
    "放进全局",
    "存进全局",
    "存为全局",
    "保存到全局",
    "转成全局",
    "转为全局",
    "加入收藏",
    "删除收藏",
    "修改昵称",
    "修改界面",
    "黑夜模式",
    "白天模式",
    "黑夜主题",
    "白天主题",
    "浅色",
    "深色",
    "界面主题",
    "更新资料",
    "修改资料",
    "班级资料",
    "个人资料",
    "移除附件",
    "删除文件",
    "删掉文件",
    "存进全局知识库",
    "放进全局库",
    "存为全局知识",
    "不要继续保存",
    "取消刚才那次保存",
    "取消保存操作",
]

DIRECT_AI_KEYWORDS = [
    "写诗",
    "写首诗",
    "写几句",
    "写一句",
    "写一段",
    "宣传口号",
    "宣传语",
    "文案",
    "鼓励",
    "打气",
    "冷笑话",
    "笑话",
    "讲故事",
    "取个名字",
    "项目名字",
    "润色",
    "翻译",
    "改写",
    "改得更顺",
    "改得更",
    "这段话改",
    "写邮件",
    "道歉邮件",
    "自我介绍",
    "演讲稿",
    "解释一下",
    "什么叫",
    "什么是",
    "讲个",
]

FILE_QA_KEYWORDS = [
    "上传",
    "刚刚上传",
    "刚才上传",
    "刚传",
    "附件",
    "文件",
    "文档",
    "PDF",
    "pdf",
    "Word",
    "word",
    "Excel",
    "excel",
    "表格",
    "课表文件",
    "课程表文件",
    "这个表格",
    "这个文档",
    "这个文件",
    "这份文档",
    "这份 PDF",
    "这张表",
]

FILE_QA_QUESTION_KEYWORDS = [
    "讲什么",
    "主要是什么",
    "主要讲",
    "总结",
    "找",
    "有没有",
    "是什么",
    "有什么",
    "写的",
    "记录了",
    "判断",
    "提到",
    "要求",
    "内容",
]

AGENT_TASK_PLAN_KEYWORDS = [
    "安排",
    "排一下",
    "规划",
    "空档",
    "空闲",
    "有空",
    "复习",
    "任务",
    "作业",
    "先做",
    "挤时间",
    "学习",
]

AGENT_TASK_KEYWORDS = [
    "安排",
    "规划",
    "待办",
    "任务",
    "空闲",
    "有空",
    "学习时间",
    "先看哪门课",
    "先写哪门课",
    "先做什么",
    "还剩哪些任务",
    "复习",
    "作业",
    "冲刺计划",
    "日程表",
    "复习计划",
]

USER_KNOWLEDGE_QA_KEYWORDS = [
    "以前保存",
    "以前存过",
    "之前保存",
    "历史上传",
    "以前导入",
    "之前导入",
    "已保存",
    "历史上传里",
    "历史上传内容",
    "全局资料",
    "全局知识库",
    "个人知识库",
    "我的知识库",
    "长期知识库",
    "保存过",
    "导入的资料",
    "保存的报名材料",
    "过去保存",
]

KNOWLEDGE_QA_PUBLIC_KEYWORDS = [
    "饭堂",
    "食堂",
    "图书馆",
    "校医院",
    "体育馆",
    "快递站",
    "宿舍网络",
    "校园卡",
    "校内",
    "学校",
    "教务处",
    "自习",
    "打印",
    "跑步",
]

UNKNOWN_EXACT_QUESTIONS = {
    "继续",
    "还有其他的吗",
    "这个呢",
    "那个呢",
    "这里呢",
    "那里呢",
    "刚才那个呢",
}

UNKNOWN_KEYWORDS = [
    "刚才",
    "刚刚",
    "上面",
    "前面",
    "上一步",
    "前一步",
    "这个",
    "那个",
    "这里",
    "那里",
    "还有其他的吗",
    "要不要",
    "是不是可以",
    "能不能换个说法",
    "是什么意思",
    "还要保留吗",
    "跳过",
]


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _is_file_qa(text: str) -> bool:
    return _contains_any(text, FILE_QA_KEYWORDS) and (
        _contains_any(text, FILE_QA_QUESTION_KEYWORDS) or not _contains_any(text, SYSTEM_ACTION_KEYWORDS)
    )


def _is_agent_schedule_plan(text: str) -> bool:
    has_schedule = "课表" in text or "课程表" in text
    has_plan = _contains_any(text, AGENT_TASK_PLAN_KEYWORDS)
    has_file_marker = _contains_any(text, ["上传", "文件", "附件", "PDF", "pdf", "Word", "word", "Excel", "excel", "表格", "文档"])
    return has_schedule and has_plan and not has_file_marker


def infer_tool_hints(question: str | None, workflow: str | None) -> list[str]:
    """Infer reserved tool hints from workflow and question text."""
    text = question or ""
    workflow = workflow or "UNKNOWN"
    tools: list[str] = []

    if _contains_any(text, TIME_KEYWORDS) or "三天" in text:
        _append_unique(tools, "GET_CURRENT_TIME")

    if workflow == "KNOWLEDGE_QA":
        _append_unique(tools, "SEARCH_PUBLIC_KNOWLEDGE")
    elif workflow == "AGENT_TASK":
        _append_unique(tools, "SEARCH_CONVERSATION_KNOWLEDGE")
        _append_unique(tools, "SEARCH_GLOBAL_KNOWLEDGE")
        if _contains_any(text, SCHEDULE_KEYWORDS) or _contains_any(text, AGENT_TASK_PLAN_KEYWORDS) or _is_agent_schedule_plan(text):
            _append_unique(tools, "SEARCH_USER_SCHEDULE")
        if _contains_any(text, TODO_KEYWORDS):
            _append_unique(tools, "SEARCH_USER_TODO")
    elif workflow == "FILE_QA":
        _append_unique(tools, "SEARCH_FILE_CONTENT")
    elif workflow == "USER_KNOWLEDGE_QA":
        _append_unique(tools, "SEARCH_GLOBAL_KNOWLEDGE")
    elif workflow == "DIRECT_AI":
        _append_unique(tools, "DIRECT_LLM")
    elif workflow == "SYSTEM_ACTION":
        return ["WRITE_USER_DATA"]
    elif workflow == "UNKNOWN":
        _append_unique(tools, "SEARCH_CONVERSATION_HISTORY")

    return tools


def _route_result(
    workflow: str,
    raw_workflow: str,
    rule_name: str | None,
    raw_confidence: float | None,
) -> RoutePostProcessResult:
    return RoutePostProcessResult(
        workflow=workflow,
        corrected_by_rule=rule_name is not None,
        rule_name=rule_name,
        raw_workflow=raw_workflow,
        raw_confidence=raw_confidence,
    )


def post_process_route(
    question: str | None,
    workflow: str | None,
    raw_confidence: float | None = None,
) -> RoutePostProcessResult:
    """Correct high-confidence boundary cases after model prediction."""
    text = (question or "").strip()
    raw_workflow = workflow or "UNKNOWN"

    if not text:
        return _route_result("UNKNOWN", raw_workflow, "UNKNOWN_EMPTY_RULE", raw_confidence)

    if _contains_any(text, SYSTEM_ACTION_KEYWORDS):
        return _route_result("SYSTEM_ACTION", raw_workflow, "SYSTEM_ACTION_RULE", raw_confidence)

    if _contains_any(text, DIRECT_AI_KEYWORDS):
        return _route_result("DIRECT_AI", raw_workflow, "DIRECT_AI_WRITING_RULE", raw_confidence)

    if _contains_any(text, USER_KNOWLEDGE_QA_KEYWORDS):
        return _route_result("USER_KNOWLEDGE_QA", raw_workflow, "USER_KNOWLEDGE_RULE", raw_confidence)

    if _is_file_qa(text):
        return _route_result("FILE_QA", raw_workflow, "FILE_QA_RULE", raw_confidence)

    if _contains_any(text, KNOWLEDGE_QA_PUBLIC_KEYWORDS):
        return _route_result("KNOWLEDGE_QA", raw_workflow, "KNOWLEDGE_QA_PUBLIC_RULE", raw_confidence)

    if text in UNKNOWN_EXACT_QUESTIONS or _contains_any(text, UNKNOWN_KEYWORDS):
        return _route_result("UNKNOWN", raw_workflow, "UNKNOWN_CONTEXT_RULE", raw_confidence)

    if _is_agent_schedule_plan(text) or _contains_any(text, AGENT_TASK_KEYWORDS):
        return _route_result("AGENT_TASK", raw_workflow, "AGENT_TASK_SCHEDULE_RULE", raw_confidence)

    return _route_result(raw_workflow, raw_workflow, None, raw_confidence)


def infer_workflow_override(question: str | None) -> str | None:
    """Backward-compatible wrapper for older callers."""
    result = post_process_route(question, None)
    if not (question or "").strip():
        return "UNKNOWN"
    return result.workflow if result.corrected_by_rule else None
