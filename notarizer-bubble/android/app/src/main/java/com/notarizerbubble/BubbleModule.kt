package com.notarizerbubble

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.provider.Settings
import android.widget.Toast
import com.facebook.react.bridge.*

class BubbleModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext), ActivityEventListener {

    private var screenshotPromise: Promise? = null
    private val REQUEST_SCREENSHOT = 1001
    private val REQUEST_OVERLAY = 1002

    override fun getName() = "BubbleModule"

    init {
        reactContext.addActivityEventListener(this)
    }

    override fun onNewIntent(intent: Intent) {}

    // Required stubs for NativeEventEmitter (even though we don't send events to JS anymore)
    @ReactMethod
    fun addListener(eventName: String) {}

    @ReactMethod
    fun removeListeners(count: Int) {}

    @ReactMethod
    fun startBubble(promise: Promise) {
        val activity = reactContext.currentActivity
        if (activity == null) {
            promise.reject("NO_ACTIVITY", "No current activity found.")
            return
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !Settings.canDrawOverlays(reactContext)) {
            val intent = Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:${reactContext.packageName}"))
            activity.startActivityForResult(intent, REQUEST_OVERLAY)
            promise.reject("NO_OVERLAY_PERMISSION", "Overlay permission not granted. User has been prompted.")
            return
        }
        requestScreenshotPermission(promise)
    }

    private fun requestScreenshotPermission(promise: Promise) {
        val activity = reactContext.currentActivity
        if (activity == null) {
            promise.reject("NO_ACTIVITY", "Activity is not available to request screenshot permission.")
            return
        }
        screenshotPromise = promise
        val projectionManager = reactContext.getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        activity.startActivityForResult(projectionManager.createScreenCaptureIntent(), REQUEST_SCREENSHOT)
    }

    override fun onActivityResult(activity: Activity, requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == REQUEST_SCREENSHOT) {
            if (resultCode == Activity.RESULT_OK && data != null) {
                Toast.makeText(reactContext, "Permission granted. Starting bubble...", Toast.LENGTH_SHORT).show()
                val serviceIntent = Intent(reactContext, BubbleService::class.java).apply {
                    putExtra("resultCode", resultCode)
                    putExtra("data", data)
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    reactContext.startForegroundService(serviceIntent)
                } else {
                    reactContext.startService(serviceIntent)
                }
                screenshotPromise?.resolve("Bubble service started successfully.")
            } else {
                screenshotPromise?.reject("SCREENSHOT_REJECTED", "Screenshot permission was denied by the user.")
            }
            screenshotPromise = null
        }
    }

    @ReactMethod
    fun stopBubble(promise: Promise) {
        val intent = Intent(reactContext, BubbleService::class.java)
        val stopped = reactContext.stopService(intent)
        promise.resolve(stopped)
    }
}