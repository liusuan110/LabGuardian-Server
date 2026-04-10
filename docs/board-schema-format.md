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

## Recommended Next Step

下一阶段建议把完整比赛板型正式写进 JSON：

- 全部主区孔位
- 电源轨分段
- 面包板物理不连续段
- 轨道标签到 `VCC / GND` 的映射
