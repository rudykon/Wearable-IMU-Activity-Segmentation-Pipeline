package com.imu.realtime

import android.content.Context
import android.content.res.Configuration
import java.util.Locale

object LocaleHelper {
    const val LANGUAGE_ZH = "zh"
    const val LANGUAGE_EN = "en"

    private const val PREFS_NAME = "app_settings"
    private const val KEY_LANGUAGE = "language"

    fun wrap(base: Context): Context {
        val tag = savedLanguageTag(base)
        val locale = localeForTag(tag)
        Locale.setDefault(locale)

        val config = Configuration(base.resources.configuration)
        config.setLocale(locale)
        return base.createConfigurationContext(config)
    }

    fun savedLanguageTag(context: Context): String {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(KEY_LANGUAGE, LANGUAGE_ZH)
            ?: LANGUAGE_ZH
    }

    fun isEnglish(context: Context): Boolean = savedLanguageTag(context) == LANGUAGE_EN

    fun nextLanguageTag(context: Context): String {
        return if (isEnglish(context)) LANGUAGE_ZH else LANGUAGE_EN
    }

    fun setLanguageTag(context: Context, tag: String) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_LANGUAGE, tag)
            .apply()
        applyToResources(context, tag)
    }

    fun applyToResources(context: Context, tag: String = savedLanguageTag(context)) {
        val locale = localeForTag(tag)
        Locale.setDefault(locale)

        val config = Configuration(context.resources.configuration)
        config.setLocale(locale)
        @Suppress("DEPRECATION")
        context.resources.updateConfiguration(config, context.resources.displayMetrics)
    }

    private fun localeForTag(tag: String): Locale {
        return if (tag == LANGUAGE_EN) Locale.ENGLISH else Locale.SIMPLIFIED_CHINESE
    }
}
