# Board Schema Format

`BoardSchema` 用来把视觉端的 `hole_id` 映射成服务器端稳定可比较的
`electrical_node_id`。

当前第一阶段已经支持：

- 默认从 [`app/domain/data/board_schemas/breadboard_legacy_v1.json`](/Users/liusuan/Desktop/LabGuardian-Server/app/domain/data/board_schemas/breadboard_legacy_v1.json) 加载
- 若 JSON 未覆盖某个孔位，则回退到代码里的默认推断规则
- 支持 `aliases` 做电源/历史命名兼容
- 支持 `generated_groups` 批量展开整块面包板孔位

## Minimal JSON Shape

```json
{
  "schema_id": "breadboard_legacy_v1",
  "board_type": "competition_breadboard_63row_dualrail",
  "aliases": {
    "VCC": "PWR_PLUS",
    "GND": "PWR_MINUS"
  },
  "generated_groups": [
    {
      "kind": "main_strip",
      "cols": ["A", "B", "C", "D", "E"],
      "row_start": 1,
      "row_end": 63,
      "side": "L",
      "electrical_node_template": "ROW_{row}_{side}"
    },
    {
      "kind": "track",
      "track": "LP",
      "segments": [
        {"row_start": 1, "row_end": 31, "suffix": "_SEG1"},
        {"row_start": 32, "row_end": 63, "suffix": "_SEG2"}
      ],
      "electrical_node_template": "TRACK_{track}{segment_suffix}"
    }
  ],
  "holes": {
    "A12": {
      "hole_id": "A12",
      "electrical_node_id": "ROW_12_L",
      "group_type": "main_grid",
      "row": 12,
      "col": "A"
    }
  }
}
```

## Field Meaning

- `schema_id`: schema 版本号，写进 `netlist_v2.board_schema_id`
- `board_type`: 板型标识
- `aliases`: 历史命名或前端简写到正式 `hole_id` 的映射
- `generated_groups`: 用规则批量生成主区孔位和电源轨孔位
- `holes`: 显式孔位覆盖表

`generated_groups[*]` 支持两类：

- `main_strip`
  - `cols`
  - `row_start`
  - `row_end`
  - `side`
  - `electrical_node_template`
- `track`
  - `track`
  - `row_start` / `row_end` 或 `segments`
  - `electrical_node_template`

`holes[*]` 字段：

- `hole_id`: 物理孔位 ID
- `electrical_node_id`: 静态导通节点 ID
- `group_type`: `main_grid / rail / power / track / custom`
- `row`: 可选，行号
- `col`: 可选，列名或轨道名

## Current Fallback Rules

如果 JSON 里没有显式给出某个孔位，当前会按默认规则推断：

- `A1-E* -> ROW_{row}_L`
- `F1-J* -> ROW_{row}_R`
- `PWR_PLUS_* -> PWR_PLUS`
- `PWR_MINUS_* -> PWR_MINUS`
- `LP* / LN* / RP* / RN* -> TRACK_{track}`
- `RAIL_*_* -> RAIL_*`

## Current Default Competition Layout

当前仓库默认 schema 已经生成：

- `A1-E63`
- `F1-J63`
- `LP1-LP31 -> TRACK_LP_SEG1`
- `LP32-LP63 -> TRACK_LP_SEG2`
- `LN1-LN31 -> TRACK_LN_SEG1`
- `LN32-LN63 -> TRACK_LN_SEG2`
- `RP1-RP31 -> TRACK_RP_SEG1`
- `RP32-RP63 -> TRACK_RP_SEG2`
- `RN1-RN31 -> TRACK_RN_SEG1`
- `RN32-RN63 -> TRACK_RN_SEG2`

同时兼容历史逻辑坐标：

- `rail_top+ -> LP`
- `rail_top- -> LN`
- `rail_bot+ -> RP`
- `rail_bot- -> RN`

## 最小用户标注

前端现在只需要让用户标注电源轨和输入/输出端口，内部 signal 不需要人工命名。
推荐请求结构：

```json
{
  "rail_assignments": {
    "top_plus": "VCC",
    "top_minus": "VCC",
    "bot_plus": "GND",
    "bot_minus": "VEE"
  },
  "port_annotations": [
    {
      "role": "input",
      "target": {"component_id": "R1", "pin_name": "pin1"},
      "label": "UI1"
    },
    {
      "role": "output",
      "target": {"hole_id": "A12"}
    }
  ]
}
```

`target` 可选择以下任一定位方式：

- `electrical_net_id`: 已知当前电气网络时直接指定。
- `component_id` + `pin_name`: 用户点中元件引脚时使用。
- `hole_id`: 用户点中面包板孔位时使用。
- `electrical_node_id`: 已知静态导通节点时使用。

`label` 可选。若不传，系统只写入 `input` / `output` 角色，具体 `UI1` / `UO1`
等端口标签由逻辑参考电路和比较阶段推断。`signal` 不再建议由前端手动标注；
电源/地仍通过 `rail_assignments` 设置。

## Future Work

默认 schema 已经覆盖当前 63 行双电源轨比赛板假设。后续工作不再是简单“补完整
JSON”，而是把实物校验、版本化和 edge 部署绑定起来：

- 用实物板确认电源轨物理分段是否确实为 `1-31 / 32-63`。
- 若比赛板存在不同批次或不同型号，新增独立 schema JSON，不覆盖
  `breadboard_legacy_v1`。
- 将正式板型的 `schema_id` 写入 `runtime_metadata` 和论文实验配置。
- 为每个 board profile 增加 smoke test，至少覆盖主区、左右侧、电源轨分段和历史
  `rail_top+ / rail_bot-` 兼容坐标。
