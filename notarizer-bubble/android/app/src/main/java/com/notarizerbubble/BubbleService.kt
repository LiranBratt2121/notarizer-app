package com.notarizerbubble

import android.app.Activity
import android.app.AlertDialog
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import android.view.*
import android.widget.ImageView
import android.widget.Toast
import androidx.core.app.NotificationCompat
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class BubbleService : Service() {

    private val VALIDATE_URL = "http://10.0.0.102:8000/validate-image"
    private val client = OkHttpClient()

    private lateinit var windowManager: WindowManager
    private lateinit var bubbleView: View
    private val CHANNEL_ID = "BubbleServiceChannel"
    private var mediaProjection: MediaProjection? = null
    private var imageReader: ImageReader? = null
    private var isActionInProgress = false
    private val mediaProjectionCallback = object : MediaProjection.Callback() {
        override fun onStop() {
            stopSelf()
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        createNotificationChannel()
        val notification: Notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Bubble Service Running").setContentText("Tap the bubble to validate").setSmallIcon(R.mipmap.ic_launcher).build()
        startForeground(1, notification)
        val resultCode = intent?.getIntExtra("resultCode", -1) ?: -1
        val resultData = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            intent?.getParcelableExtra("data", Intent::class.java)
        } else { @Suppress("DEPRECATION") intent?.getParcelableExtra("data") }
        if (resultCode == Activity.RESULT_OK && resultData != null) {
            val projectionManager = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            mediaProjection = projectionManager.getMediaProjection(resultCode, resultData)
            mediaProjection?.registerCallback(mediaProjectionCallback, Handler(Looper.getMainLooper()))
            setupImageReader()
        }
        return START_STICKY
    }

    override fun onCreate() {
        super.onCreate()
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val inflater = getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        bubbleView = inflater.inflate(R.layout.bubble_layout, null)
        val layoutParams = WindowManager.LayoutParams(WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.WRAP_CONTENT, if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY else WindowManager.LayoutParams.TYPE_PHONE, WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE, PixelFormat.TRANSLUCENT)
        layoutParams.gravity = Gravity.TOP or Gravity.START
        layoutParams.x = 100; layoutParams.y = 100
        bubbleView.setOnTouchListener(BubbleTouchListener(layoutParams, this::showValidationMenu))
        windowManager.addView(bubbleView, layoutParams)
    }

    private fun showValidationMenu() {
        Handler(Looper.getMainLooper()).post {
            val menuOptions = arrayOf<CharSequence>("Validate Screen", "Close Bubble")
            val builder = AlertDialog.Builder(this).setTitle("Bubble Menu")
            builder.setItems(menuOptions) { dialog, item ->
                when (item) {
                    0 -> {
                        // --- THIS IS THE FIX ---
                        // 1. Dismiss the dialog immediately.
                        dialog.dismiss()
                        // 2. Wait a short moment for the dialog to disappear from the screen.
                        Handler(Looper.getMainLooper()).postDelayed({
                            // 3. NOW, start the screenshot process.
                            takeScreenshotAndValidate()
                        }, 200) // 200ms delay is usually enough.
                    }
                    1 -> stopSelf()
                }
                // We no longer need to dismiss here as it's handled above.
            }
            val dialog = builder.create()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                dialog.window?.setType(WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY)
            } else { @Suppress("DEPRECATION") dialog.window?.setType(WindowManager.LayoutParams.TYPE_SYSTEM_ALERT) }
            dialog.show()
        }
    }

    private fun takeScreenshotAndValidate() {
        if (isActionInProgress) return
        isActionInProgress = true
        showToast("Capturing...")
        bubbleView.visibility = View.INVISIBLE
        Handler(Looper.getMainLooper()).postDelayed({
            val image = imageReader?.acquireLatestImage() ?: run {
                isActionInProgress = false
                bubbleView.visibility = View.VISIBLE
                return@postDelayed
            }
            val planes = image.planes; val buffer = planes[0].buffer
            val pixelStride = planes[0].pixelStride; val rowStride = planes[0].rowStride
            val rowPadding = rowStride - pixelStride * image.width
            val bitmap = Bitmap.createBitmap(image.width + rowPadding / pixelStride, image.height, Bitmap.Config.ARGB_8888)
            bitmap.copyPixelsFromBuffer(buffer)
            image.close()
            bubbleView.visibility = View.VISIBLE
            val file = File(cacheDir, "screenshot-to-validate.jpg")
            try {
                FileOutputStream(file).use { out -> bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out) }
                uploadForValidation(file)
            } catch (e: Exception) {
                e.printStackTrace()
                isActionInProgress = false
            }
        }, 300)
    }

    private fun uploadForValidation(file: File) {
        showToast("Validating with server...")
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("file", file.name, file.asRequestBody("image/jpeg".toMediaType()))
            .build()
        val request = Request.Builder().url(VALIDATE_URL).post(requestBody).build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                e.printStackTrace()
                showResultDialog("Validation Failed", "Could not connect to the server: ${e.message}")
                isActionInProgress = false
            }
            override fun onResponse(call: Call, response: Response) {
                response.use {
                    val responseBody = response.body?.string()
                    if (!response.isSuccessful) {
                        showResultDialog("Validation Failed", "Server returned an error (Code: ${response.code}).")
                    } else {
                        Log.d("BubbleService", "Server Response: $responseBody")
                        try {
                            val json = JSONObject(responseBody)
                            val isValid = json.getBoolean("is_valid")
                            if (isValid) {
                                showResultDialog("Validation Success", "The image is valid!")
                            } else {
                                showResultDialog("Validation Failed", "The image is not recognized.")
                            }
                        } catch (e: Exception) {
                            showResultDialog("Error", "Could not understand the server's response.")
                        }
                    }
                    isActionInProgress = false
                }
            }
        })
    }

    private fun showResultDialog(title: String, message: String) {
        Handler(Looper.getMainLooper()).post {
            val builder = AlertDialog.Builder(this)
            builder.setTitle(title)
            builder.setMessage(message)
            builder.setPositiveButton("OK") { dialog, _ ->
                dialog.dismiss()
            }
            val dialog = builder.create()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                dialog.window?.setType(WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY)
            } else { @Suppress("DEPRECATION") dialog.window?.setType(WindowManager.LayoutParams.TYPE_SYSTEM_ALERT) }
            dialog.show()
        }
    }

    private fun showToast(message: String) {
        Handler(Looper.getMainLooper()).post { Toast.makeText(this, message, Toast.LENGTH_SHORT).show() }
    }

    inner class BubbleTouchListener(private val params: WindowManager.LayoutParams, private val onClick: () -> Unit) :
        View.OnTouchListener {
        private var initialX = 0; private var initialY = 0; private var initialTouchX = 0f; private var initialTouchY = 0f
        private val CLICK_ACTION_THRESHOLD = 10
        override fun onTouch(v: View, event: MotionEvent): Boolean {
            when (event.action) {
                MotionEvent.ACTION_DOWN -> { initialX = params.x; initialY = params.y; initialTouchX = event.rawX; initialTouchY = event.rawY; return true }
                MotionEvent.ACTION_UP -> { if (Math.abs((event.rawX - initialTouchX).toInt()) < CLICK_ACTION_THRESHOLD && Math.abs((event.rawY - initialTouchY).toInt()) < CLICK_ACTION_THRESHOLD) { onClick() }; return true }
                MotionEvent.ACTION_MOVE -> { params.x = initialX + (event.rawX - initialTouchX).toInt(); params.y =
                    initialY + (event.rawY - initialTouchY).toInt(); windowManager.updateViewLayout(bubbleView, params); return true }
            }
            return false
        }
    }

    private fun setupImageReader() {
        val metrics = resources.displayMetrics
        val density = metrics.densityDpi; val width = metrics.widthPixels; val height = metrics.heightPixels
        imageReader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2)
        mediaProjection?.createVirtualDisplay(
            "ScreenCapture", width, height, density,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            imageReader?.surface, null, Handler(Looper.getMainLooper())
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        if (this::bubbleView.isInitialized) {
            windowManager.removeView(bubbleView)
        }
        mediaProjection?.unregisterCallback(mediaProjectionCallback)
        mediaProjection?.stop()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val serviceChannel = NotificationChannel(
                CHANNEL_ID,
                "Bubble Service Channel",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java); manager.createNotificationChannel(
                serviceChannel
            )
        }
    }
}