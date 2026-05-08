package com.imu.realtime

import android.content.Context
import android.util.AttributeSet
import com.google.android.material.bottomnavigation.BottomNavigationView

/**
 * BottomNavigationView subclass that raises the item limit from 5 to 6.
 * The default cap (5) triggers IllegalArgumentException when inflating a 6-item menu.
 */
class WideBottomNavView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = com.google.android.material.R.attr.bottomNavigationStyle
) : BottomNavigationView(context, attrs, defStyleAttr) {

    override fun getMaxItemCount(): Int = 6
}
