package com.imu.realtime

/**
 * 左手骨骼模型 —— 自然下垂姿态（零旋转时展示）
 *
 * 坐标系（cm）：
 *   +X = 朝拇指方向（正面看左手时，向右）
 *   +Y = 朝肩膀方向（上）
 *   +Z = 朝手掌正面（朝向观察者）
 *
 * 零旋转时，在屏幕上呈现：
 *   手臂从腕部向上延伸（肘部在上）
 *   手指自然下垂微卷（在下）
 *   手掌朝向观察者
 *
 * 关节编号：
 *   0: 手臂远端（手腕上方 8 cm，朝肘部）
 *   1: 手腕
 *   2-5:  拇指 CMC/MCP/DIP/TIP
 *   6-9:  食指 MCP/PIP/DIP/TIP
 *   10-13: 中指
 *   14-17: 无名指
 *   18-21: 小拇指
 */
object HandModel {

    data class Joint(val x: Float, val y: Float, val z: Float)
    data class Bone(val from: Int, val to: Int, val widthCm: Float)

    val joints: List<Joint> = listOf(
        // 0: 手臂远端（肘部方向，在腕部上方 8 cm）
        Joint( 0.0f,  8.0f,  0.0f),
        // 1: 手腕
        Joint( 0.0f,  0.0f,  0.0f),

        // 2-5: 拇指（左手拇指向 +X 方向，且稍微向观察者方向 +Z）
        Joint( 2.2f, -0.6f,  0.4f),   // 2 拇指 CMC
        Joint( 3.6f, -1.7f,  0.9f),   // 3 拇指 MCP
        Joint( 4.6f, -2.7f,  1.2f),   // 4 拇指 DIP
        Joint( 5.2f, -3.5f,  1.3f),   // 5 拇指 TIP

        // 6-9: 食指（微向拇指侧偏，自然微卷向 +Z）
        Joint( 1.7f, -2.0f,  0.2f),   // 6 食指 MCP
        Joint( 1.7f, -4.5f,  0.7f),   // 7 食指 PIP（自然卷曲）
        Joint( 1.5f, -6.2f,  1.1f),   // 8 食指 DIP
        Joint( 1.4f, -7.6f,  1.3f),   // 9 食指 TIP

        // 10-13: 中指（最长）
        Joint( 0.3f, -2.1f,  0.2f),   // 10 中指 MCP
        Joint( 0.3f, -5.0f,  0.7f),   // 11 中指 PIP
        Joint( 0.2f, -6.8f,  1.2f),   // 12 中指 DIP
        Joint( 0.1f, -8.3f,  1.4f),   // 13 中指 TIP

        // 14-17: 无名指
        Joint(-0.9f, -2.0f,  0.2f),   // 14 无名指 MCP
        Joint(-1.0f, -4.5f,  0.7f),   // 15 无名指 PIP
        Joint(-1.1f, -6.1f,  1.1f),   // 16 无名指 DIP
        Joint(-1.1f, -7.5f,  1.3f),   // 17 无名指 TIP

        // 18-21: 小拇指（稍短）
        Joint(-2.1f, -1.8f,  0.2f),   // 18 小拇指 MCP
        Joint(-2.2f, -3.9f,  0.6f),   // 19 小拇指 PIP
        Joint(-2.3f, -5.2f,  0.9f),   // 20 小拇指 DIP
        Joint(-2.3f, -6.2f,  1.1f)    // 21 小拇指 TIP
    )

    /** 骨骼（widthCm = 截面直径，越近掌越粗） */
    val bones: List<Bone> = listOf(
        // 手臂
        Bone(0, 1, 3.4f),
        // 掌骨（腕到各 MCP）
        Bone(1,  2, 1.9f),
        Bone(1,  6, 2.0f),
        Bone(1, 10, 2.1f),
        Bone(1, 14, 2.0f),
        Bone(1, 18, 1.7f),
        // 掌横韧带
        Bone( 6, 10, 1.8f),
        Bone(10, 14, 1.8f),
        Bone(14, 18, 1.6f),
        // 拇指
        Bone(2, 3, 1.8f), Bone(3, 4, 1.5f), Bone(4, 5, 1.1f),
        // 食指
        Bone(6, 7, 1.6f), Bone(7, 8, 1.3f), Bone(8, 9, 1.0f),
        // 中指
        Bone(10, 11, 1.7f), Bone(11, 12, 1.4f), Bone(12, 13, 1.1f),
        // 无名指
        Bone(14, 15, 1.5f), Bone(15, 16, 1.2f), Bone(16, 17, 1.0f),
        // 小拇指
        Bone(18, 19, 1.2f), Bone(19, 20, 1.0f), Bone(20, 21, 0.8f)
    )

    /** 手掌多边形顶点索引（腕 + 各MCP） */
    val palmFace: List<Int> = listOf(1, 2, 6, 10, 14, 18)
}
