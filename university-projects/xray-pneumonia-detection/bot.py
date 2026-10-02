"""
Telegram bot for chest X-ray classification with LIME interpretation + X-ray detection.
"""

import os
import io
import asyncio
import numpy as np
from PIL import Image, ImageStat
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, CallbackQueryHandler, filters
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from concurrent.futures import ThreadPoolExecutor

import config
from image_check import is_xray_image
from interpret import generate_lime_image

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store the last uploaded image
LAST_IMAGE = None

# ThreadPoolExecutor for async LIME
executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------
# HELPER FUNCTIONS FOR ENGAGEMENT
# ---------------------------
def get_certainty_emoji(confidence: float) -> str:
    """Translates model confidence into an engaging certainty rating."""
    if confidence >= 0.95:
        return "🔥 High Certainty"
    elif confidence >= 0.85:
        return "👍 Strong Likelihood"
    elif confidence >= 0.70:
        return "🤔 Moderate Concern"
    else:
        return "🧐 Could be either"


# ---------------------------
# LOAD MODEL
# ---------------------------
def load_first_model():
    model_path = os.path.join(config.SAVED_MODELS_DIR, config.BOT_MODEL_FILE)
    if not os.path.exists(model_path):
        models_list = sorted(f for f in os.listdir(config.SAVED_MODELS_DIR) if f.endswith(".h5"))
        if not models_list:
            raise ValueError("❌ No saved models in saved_models/. Train and save models first.")
        model_path = os.path.join(config.SAVED_MODELS_DIR, models_list[0])
    print(f"Loading model: {model_path}")
    return load_model(model_path, safe_mode=False), model_path


MODEL, MODEL_PATH = load_first_model()


# ---------------------------
# /START
# ---------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📸 Send a photo", callback_data="send_photo")],
        [InlineKeyboardButton("ℹ️ Help", callback_data="help")]
    ]

    await update.message.reply_text(
        "👋 Hey there! I'm **Dr. XRayDetectBot** 🫁, your AI assistant for chest X-rays.\n\n"
        "Send me a clear X-ray photo, and I'll quickly check for **NORMAL** or **PNEUMONIA**.\n\n"
        "💡 Best Part: I'll show you *exactly* what the AI saw using my powerful **LIME interpretation**! 🧠",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )


# ---------------------------
# /HELP
# ---------------------------
async def help_command(update, context):
    if hasattr(update, "message") and update.message:
        target = update.message
    elif hasattr(update, "callback_query") and update.callback_query:
        target = update.callback_query.message
    else:
        return

    await target.reply_text(
        "🩻 How to use the bot:\n\n"
        "1️⃣ Send a chest X-ray image\n"
        "2️⃣ Receive the classification result and certainty score\n"
        "3️⃣ Press 'Interpretation' to see the LIME heatmap (the AI's 'reasoning')\n\n"
        "💡 For best results, send the image as a *file*, not a compressed photo.",
        parse_mode="Markdown"
    )


# ---------------------------
# HANDLE IMAGE UPLOAD
# ---------------------------
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LAST_IMAGE
    message = update.message

    file = None
    if message.photo:
        file = await message.photo[-1].get_file()
    elif message.document and message.document.mime_type.startswith("image/"):
        file = await message.document.get_file()
    else:
        await message.reply_text("⚠️ Please send an image file.")
        return

    # Read image
    bio = io.BytesIO()
    await file.download_to_memory(out=bio)
    bio.seek(0)
    image = Image.open(bio).convert("RGB")

    # X-RAY CHECK
    if not is_xray_image(image):
        await message.reply_text(
            "🧐 Hmm... I don't think that's a chest X-ray, or it might be too blurry/dark for me to read.\n\n"
            "**Please send a clear, uncropped image of a chest X-ray.** Let's try again!"
        )
        return

    LAST_IMAGE = image

    # Prepare input
    arr = img_to_array(image.resize(config.IMG_SIZE)) / 255.0
    arr = np.expand_dims(arr, axis=0)

    proba = float(MODEL.predict(arr)[0][0])

    # Determine result and confidence for display
    if proba >= config.PREDICTION_THRESHOLD:
        label = "🩸 *PNEUMONIA*"
        conf = proba
    else:
        label = "💨 *NORMAL*"
        conf = 1 - proba

    certainty_text = get_certainty_emoji(conf)  # Use engaging certainty text

    keyboard = [
        [
            InlineKeyboardButton("📸 Try another", callback_data="send_photo"),
            InlineKeyboardButton("🧠 Interpretation (LIME)", callback_data="interpret")
        ]
    ]

    await message.reply_text(
        f"✅ **AI Scan Complete!**\n\n"
        f"**Result:** {label} ({certainty_text})\n"
        f"**Certainty:** *{conf * 100:.2f}%*\n"
        f"⚙️ Model: `{os.path.basename(MODEL_PATH)}`\n\n"
        f"Curious what the AI is focusing on? Tap **Interpretation** below! ⬇️",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


# ---------------------------
# INTERPRETATION BUTTON HANDLER
# ---------------------------
async def interpret_image(update, context):
    global LAST_IMAGE
    query = update.callback_query
    await query.answer()

    if LAST_IMAGE is None:
        await query.message.reply_text("⚠️ No image found. Send an X-ray first.")
        return

    # Step 1: Inform user with engaging message
    msg = await query.message.reply_text(
        "🔬 **Unlocking the AI's Brain...** This might take a moment. "
        "The LIME analysis is checking 1000 versions of your X-ray to find the key areas!"
    )

    # -------------------------
    # LIME
    # -------------------------
    try:
        lime_img = await asyncio.get_event_loop().run_in_executor(
            executor, generate_lime_image, LAST_IMAGE, MODEL
        )
        bio_lime = io.BytesIO()
        lime_img.save(bio_lime, format="PNG")
        bio_lime.seek(0)

        await query.message.reply_photo(
            photo=bio_lime,
            caption="💡 **LIME Spotlight:** See the areas the model focused on to make its decision! (Highlighted regions strongly influenced the prediction)"
        )
        await msg.delete()

    except Exception as e:
        await query.message.reply_text(f"⚠️ LIME error: {e}")
        await msg.delete()

    # ---------------------------


# INLINE BUTTON HANDLER
# ---------------------------
async def button_handler(update, context):
    query = update.callback_query
    await query.answer()

    if query.data == "send_photo":
        await query.message.reply_text("📤 Please send a chest X-ray image.")
    elif query.data == "help":
        await help_command(update, context)
    elif query.data == "interpret":
        await interpret_image(update, context)


# ---------------------------
# ERROR HANDLER
# ---------------------------
async def error_handler(update, context):
    logger.error(msg="Error while processing update", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text("⚠️ Something went wrong. Try again.")


# ---------------------------
# SET BOT COMMANDS
# ---------------------------
async def set_commands(application):
    commands = [
        BotCommand("start", "Start using Dr. XRayDetectBot"),
        BotCommand("help", "How to get a diagnosis"),
    ]
    await application.bot.set_my_commands(commands)


# ---------------------------
# MAIN ENTRY POINT
# ---------------------------
def main():
    token = config.TELEGRAM_BOT_TOKEN
    if not token or token == "YOUR_TELEGRAM_BOT_TOKEN":
        raise ValueError("❌ TELEGRAM_BOT_TOKEN environment variable is not set")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_error_handler(error_handler)
    app.post_init = set_commands

    print("🤖 Dr. XRayDetectBot started. Waiting for X-rays...")
    app.run_polling()


if __name__ == "__main__":
    main()