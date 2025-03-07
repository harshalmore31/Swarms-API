# Swarms API 

轻松构建、部署和编排大规模AI代理。Swarms API提供了全面的端点套件，用于创建和管理多代理系统。

## 功能

- **Swarms API**：一个强大的REST API，用于轻松管理和执行多代理系统。
- **灵活的模型支持**：利用各种AI模型，包括GPT-4o、Claude、Deepseek和根据您需求定制的模型。
- **多样化的群体架构**：选择多种群体架构，如并发、顺序和混合工作流，以优化任务执行。
- **动态代理配置**：轻松配置代理，为不同角色和任务自定义参数。
- **Supabase集成**：内置数据库支持，用于日志记录、API密钥管理和用户认证。
- **实时监控**：实时跟踪群体性能和执行指标，以获得更好的洞察和调整。
- **批处理**：同时执行多个群体任务，提高效率和吞吐量。
- **任务调度**：安排未来时间的群体执行，自动化重复任务。
- **任务调度**：安排未来时间的群体执行，自动化重复任务。
- **使用跟踪**：监控API使用情况和积分消耗。

## API文档和资源

- **文档**：[Swarms API文档](https://docs.swarms.world/en/latest/swarms_cloud/swarms_api/)
- **价格信息**：[API价格](https://docs.swarms.world/en/latest/swarms_cloud/api_pricing/)
- **API密钥**：[获取API密钥](https://swarms.world/platform/api-keys)

## API端点

### 核心端点

| 端点 | 方法 | 描述 | 参数 |
|----------|--------|-------------|------------|
| `/health` | GET | 检查API健康状态 | 无 |
| `/v1/swarms/available` | GET | 列出可用的群体类型 | 无 |

### 群体操作端点

| 端点 | 方法 | 描述 | 参数 |
|----------|--------|-------------|------------|
| `/v1/swarm/completions` | POST | 运行单个群体任务 | `SwarmSpec`对象 |
| `/v1/swarm/batch/completions` | POST | 运行多个群体任务 | `SwarmSpec`对象数组 |
| `/v1/swarm/logs` | GET | 检索API请求日志 | 无 |

### 调度端点

| 端点 | 方法 | 描述 | 参数 |
|----------|--------|-------------|------------|
| `/v1/swarm/schedule` | POST | 调度群体任务 | 带有`schedule`对象的`SwarmSpec` |
| `/v1/swarm/schedule` | GET | 列出所有计划任务 | 无 |
| `/v1/swarm/schedule/{job_id}` | DELETE | 取消计划任务 | `job_id` |

## 请求参数

### SwarmSpec对象

| 参数 | 类型 | 必需 | 描述 |
|-----------|------|----------|-------------|
| `name` | 字符串 | 否 | 群体名称 |
| `description` | 字符串 | 否 | 群体目的的描述 |
| `agents` | 数组 | 是 | 代理配置数组 |
| `max_loops` | 整数 | 否 | 最大迭代循环数（默认：1） |
| `swarm_type` | 字符串 | 否 | 工作流类型（"ConcurrentWorkflow"、"SequentialWorkflow"等） |
| `rearrange_flow` | 字符串 | 否 | 重新安排工作流的指令 |
| `task` | 字符串 | 是 | 要执行的任务 |
| `img` | 字符串 | 否 | 可选图像URL |
| `return_history` | 布尔值 | 否 | 包含对话历史（默认：true） |
| `rules` | 字符串 | 否 | 代理行为指南 |
| `schedule` | 对象 | 否 | 调度详情（适用于计划任务） |

### AgentSpec对象

| 参数 | 类型 | 必需 | 描述 |
|-----------|------|----------|-------------|
| `agent_name` | 字符串 | 是 | 代理名称 |
| `description` | 字符串 | 否 | 代理目的描述 |
| `system_prompt` | 字符串 | 否 | 代理的系统提示 |
| `model_name` | 字符串 | 是 | 要使用的AI模型（例如，"gpt-4o"、"claude-3-opus"） |
| `auto_generate_prompt` | 布尔值 | 否 | 自动生成提示 |
| `max_tokens` | 整数 | 否 | 响应的最大令牌数（默认：8192） |
| `temperature` | 浮点数 | 否 | 响应随机性（默认：0.5） |
| `role` | 字符串 | 否 | 代理角色（默认："worker"） |
| `max_loops` | 整数 | 否 | 此代理的最大循环数（默认：1） |

### ScheduleSpec对象

| 参数 | 类型 | 必需 | 描述 |
|-----------|------|----------|-------------|
| `scheduled_time` | 日期时间 | 是 | 执行任务的时间（UTC） |
| `timezone` | 字符串 | 否 | 调度的时区（默认："UTC"） |

## 群体类型

- `AgentRearrange`
- `MixtureOfAgents`
- `SpreadSheetSwarm`
- `SequentialWorkflow`
- `ConcurrentWorkflow`
- `GroupChat`
- `MultiAgentRouter`
- `AutoSwarmBuilder`
- `HiearchicalSwarm`
- `auto`
- `MajorityVoting`

## 认证

所有API端点（健康检查除外）都需要在`x-api-key`头中传递API密钥：

```bash
curl -H "x-api-key: your_api_key" -H "Content-Type: application/json" -X POST https://swarms-api-285321057562.us-east1.run.app/v1/swarm/completions
```

## 使用示例

以下是运行具有多个代理的群体的基本示例：

```python
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "https://swarms-api-285321057562.us-east1.run.app"

headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

def run_single_swarm():
    payload = {
        "name": "Financial Analysis Swarm",
        "description": "Market analysis swarm",
        "agents": [
            {
                "agent_name": "Market Analyst",
                "description": "Analyzes market trends",
                "system_prompt": "You are a financial analyst expert.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.5
            },
            {
                "agent_name": "Economic Forecaster",
                "description": "Predicts economic trends",
                "system_prompt": "You are an expert in economic forecasting.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.7
            },
            {
                "agent_name": "Data Scientist",
                "description": "Performs data analysis",
                "system_prompt": "You are a data science expert.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.3
            },
        ],
        "max_loops": 1,
        "swarm_type": "ConcurrentWorkflow",
        "task": "What are the best ETFs and index funds for AI and tech?",
        "return_history": True,
    }

    response = requests.post(
        f"{BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=payload,
    )

    return json.dumps(response.json(), indent=4)

if __name__ == "__main__":
    result = run_single_swarm()
    print("Swarm Result:")
    print(result)
```

## 调度示例

```python
import datetime
import pytz
from datetime import timedelta

# Schedule a swarm to run in 1 hour
future_time = datetime.datetime.now(pytz.UTC) + timedelta(hours=1)

schedule_payload = {
    "name": "Daily Market Analysis",
    "agents": [
        {
            "agent_name": "Market Analyzer",
            "model_name": "gpt-4o",
            "system_prompt": "You analyze financial markets daily"
        }
    ],
    "swarm_type": "SequentialWorkflow",
    "task": "Provide a summary of today's market movements",
    "schedule": {
        "scheduled_time": future_time.isoformat(),
        "timezone": "America/New_York"
    }
}

response = requests.post(
    f"{BASE_URL}/v1/swarm/schedule",
    headers=headers,
    json=schedule_payload
)
```

## 积分使用

API使用根据以下因素消耗积分：
1. 使用的代理数量
2. 输入/输出令牌数
3. 模型选择
4. 一天中的时间（非高峰时段折扣）

有关详细价格信息，请访问[API价格](https://docs.swarms.world/en/latest/swarms_cloud/api_pricing/)页面。

## 错误处理

常见HTTP状态码：
- `200`：成功
- `400`：错误请求（无效参数）
- `401`：未授权（无效或缺少API密钥）
- `402`：需要付款（积分不足）
- `429`：超过速率限制
- `500`：内部服务器错误

## 获取支持

如有问题或需要支持：
- 查看[文档](https://docs.swarms.world/en/latest/swarms_cloud/swarms_api/)
- 价格文档：[价格文档](https://docs.swarms.world/en/latest/swarms_cloud/api_pricing/)
- 联系kye@swarms.world

# API优化和功能添加待办事项列表

## 性能优化
- [ ] 为Supabase客户端实现连接池
- [ ] 增加API密钥验证函数的LRU缓存大小
- [ ] 为令牌计数操作添加缓存
- [ ] 实现API请求的批量日志记录
- [ ] 创建deduct_credits函数的异步版本
- [ ] 使用ThreadPoolExecutor进行并行代理创建
- [ ] 优化Uvicorn服务器设置（工作者、循环、并发限制）
- [ ] 在生产环境中禁用调试模式
- [ ] 添加uvloop以加快事件循环处理
- [ ] 实现数据库操作的请求批处理

## 新功能
- [ ] 添加带有详细系统统计信息的健康监控端点
- [ ] 实现用户配额管理系统
- [ ] 创建API密钥轮换功能
- [ ] 添加对代理模板/预设的支持
- [ ] 实现已完成群体任务的webhook通知
- [ ] 添加对长时间运行的群体任务的支持，并提供状态更新
- [ ] 为常用群体配置创建缓存层
- [ ] 根据用户层级实现速率限制
- [ ] 添加对自定义工具集成的支持
- [ ] 为高负载场景创建作业队列系统

## 安全增强
- [ ] 实现API密钥范围限定（只读、写入、管理员）
- [ ] 添加请求签名以增强安全性
- [ ] 实现基于IP的访问控制
- [ ] 为安全敏感操作创建审计日志
- [ ] 为日志和数据库中的敏感数据添加加密
- [ ] 实现自动可疑活动检测

## 监控和可观察性
- [ ] 添加详细的性能指标收集
- [ ] 实现带有关联ID的结构化日志记录
- [ ] 创建实时API使用监控仪表板
- [ ] 添加系统问题和异常警报
- [ ] 实现请求流的分布式跟踪
- [ ] 创建定期性能报告

## 开发者体验
- [ ] 添加带有示例的全面API文档
- [ ] 为常用编程语言创建SDK库
- [ ] 实现游乐场/测试环境
- [ ] 添加带有详细错误消息的请求/响应验证
- [ ] 创建交互式API浏览器
- [ ] 实现版本化的API端点

## 数据库优化
- [ ] 为高容量表添加数据库查询优化
- [ ] 为日志和指标实现数据库分片
- [ ] 为旧日志条目创建自动清理
- [ ] 添加数据库连接重试逻辑
- [ ] 为扩展读取操作实现读取副本

## 可靠性改进
- [ ] 为外部依赖项添加断路器模式
- [ ] 为非关键功能实现优雅降级
- [ ] 创建自动备份和恢复程序
- [ ] 为瞬态故障添加重试逻辑
- [ ] 为关键操作实现后备机制

## 多模态处理
- [ ] 实现语音到文本转换，用于音频输入处理
- [ ] 添加文本到语音功能，用于语音响应生成
- [ ] 创建图像分析和处理管道，用于视觉输入
- [ ] 开发视频处理功能，用于时间视觉数据
- [ ] 实现文档解析和提取，适用于PDF、DOC等
- [ ] 添加OCR功能，用于从图像中提取文本
- [ ] 创建多模态代理功能（结合文本、图像、音频）
- [ ] 实现不同数据类型之间的跨模态推理
- [ ] 添加从文本描述生成图像的支持
- [ ] 开发视频摘要和分析功能
