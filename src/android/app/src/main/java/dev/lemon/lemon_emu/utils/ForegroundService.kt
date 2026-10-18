// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package dev.lemon.lemon_emu.utils

import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import dev.lemon.lemon_emu.R
import dev.lemon.lemon_emu.activities.EmulationActivity

/**
 * A service that shows a permanent notification in the background to avoid the app getting
 * cleared from memory by the system.
 */
class ForegroundService : Service() {
    companion object {
        const val EMULATION_RUNNING_NOTIFICATION = 0x1000

        const val ACTION_STOP = "stop"
    }

    /** Returns false when Android does not let the service run in the foreground right now. */
    private fun showRunningNotification(): Boolean {
        // Intent is used to resume emulation if the notification is clicked
        val contentIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, EmulationActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        val builder =
            NotificationCompat.Builder(this, getString(R.string.app_notification_channel_id))
                .setSmallIcon(R.drawable.ic_stat_notification_logo)
                .setContentTitle(getString(R.string.app_name))
                .setContentText(getString(R.string.app_notification_running))
                .setPriority(NotificationCompat.PRIORITY_DEFAULT)
                .setOngoing(true)
                .setVibrate(null)
                .setSound(null)
                .setContentIntent(contentIntent)
        // Only allowed while Lemon is in the foreground (Android 12+): a restart by the system or a
        // start racing the app going to the background throws, which used to crash the app.
        return try {
            startForeground(EMULATION_RUNNING_NOTIFICATION, builder.build())
            true
        } catch (e: Exception) {
            Log.warning("[ForegroundService] Could not run in the foreground: ${e.message}")
            false
        }
    }

    override fun onBind(intent: Intent): IBinder? {
        return null
    }

    override fun onCreate() {
        if (!showRunningNotification()) {
            stopSelf()
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            try {
                NotificationManagerCompat.from(this).cancel(EMULATION_RUNNING_NOTIFICATION)
                stopForeground(STOP_FOREGROUND_REMOVE)
            } catch (e: Exception) {
                Log.error("Failed to stop foreground service")
            }
            stopSelfResult(startId)
            return START_NOT_STICKY
        }

        if (intent != null && !showRunningNotification()) {
            stopSelfResult(startId)
        }
        // Not sticky: after the process dies there is no emulation left to keep alive, and a
        // restart by the system happens in the background, where it is not allowed to start.
        return START_NOT_STICKY
    }

    override fun onDestroy() =
        NotificationManagerCompat.from(this).cancel(EMULATION_RUNNING_NOTIFICATION)
}
