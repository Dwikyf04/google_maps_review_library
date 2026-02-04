{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9b56567a-8ad1-427e-ac1e-8df29a09159d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-10-23 20:39:43.000 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-23 20:39:43.595 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\dwiky\\miniforge3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-10-23 20:39:43.595 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-23 20:39:43.595 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-23 20:39:43.595 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-23 20:39:44.107 Thread 'Thread-3': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-23 20:39:44.107 Thread 'Thread-3': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-23 20:39:44.107 Thread 'Thread-3': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "C:\\Users\\dwiky\\miniforge3\\Lib\\site-packages\\sklearn\\base.py:442: InconsistentVersionWarning: Trying to unpickle estimator TfidfTransformer from version 1.6.1 when using version 1.7.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:\n",
      "https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations\n",
      "  warnings.warn(\n",
      "C:\\Users\\dwiky\\miniforge3\\Lib\\site-packages\\sklearn\\base.py:442: InconsistentVersionWarning: Trying to unpickle estimator TfidfVectorizer from version 1.6.1 when using version 1.7.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:\n",
      "https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations\n",
      "  warnings.warn(\n",
      "2025-10-23 20:39:46.057 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-23 20:39:46.057 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-10-23 20:39:46.057 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: 'C:\\\\Users\\\\dwiky\\\\Documents\\\\googlemaps_projectt\\\\Models\\\\sentiment_model.pkl'",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mFileNotFoundError\u001b[39m                         Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[1]\u001b[39m\u001b[32m, line 12\u001b[39m\n\u001b[32m      9\u001b[39m     model = joblib.load(\u001b[33mr\u001b[39m\u001b[33m\"\u001b[39m\u001b[33mC:\u001b[39m\u001b[33m\\\u001b[39m\u001b[33mUsers\u001b[39m\u001b[33m\\\u001b[39m\u001b[33mdwiky\u001b[39m\u001b[33m\\\u001b[39m\u001b[33mDocuments\u001b[39m\u001b[33m\\\u001b[39m\u001b[33mgooglemaps_projectt\u001b[39m\u001b[33m\\\u001b[39m\u001b[33mModels\u001b[39m\u001b[33m\\\u001b[39m\u001b[33msentiment_model.pkl\u001b[39m\u001b[33m\"\u001b[39m)\n\u001b[32m     10\u001b[39m     \u001b[38;5;28;01mreturn\u001b[39;00m tfidf, model\n\u001b[32m---> \u001b[39m\u001b[32m12\u001b[39m tfidf, model = \u001b[43mload_model\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[32m     14\u001b[39m \u001b[38;5;66;03m# === UI Aplikasi ===\u001b[39;00m\n\u001b[32m     15\u001b[39m st.title(\u001b[33m\"\u001b[39m\u001b[33m📊 Analisis Sentimen Ulasan Google Maps - Perpustakaan Nasional\u001b[39m\u001b[33m\"\u001b[39m)\n",
      "\u001b[36mFile \u001b[39m\u001b[32m~\\miniforge3\\Lib\\site-packages\\streamlit\\runtime\\caching\\cache_utils.py:227\u001b[39m, in \u001b[36mCachedFunc.__call__\u001b[39m\u001b[34m(self, *args, **kwargs)\u001b[39m\n\u001b[32m    224\u001b[39m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[32m    225\u001b[39m         spinner_message = \u001b[33mf\u001b[39m\u001b[33m\"\u001b[39m\u001b[33mRunning `\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mname\u001b[38;5;132;01m}\u001b[39;00m\u001b[33m(...)`.\u001b[39m\u001b[33m\"\u001b[39m\n\u001b[32m--> \u001b[39m\u001b[32m227\u001b[39m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[43m.\u001b[49m\u001b[43m_get_or_create_cached_value\u001b[49m\u001b[43m(\u001b[49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mkwargs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mspinner_message\u001b[49m\u001b[43m)\u001b[49m\n",
      "\u001b[36mFile \u001b[39m\u001b[32m~\\miniforge3\\Lib\\site-packages\\streamlit\\runtime\\caching\\cache_utils.py:269\u001b[39m, in \u001b[36mCachedFunc._get_or_create_cached_value\u001b[39m\u001b[34m(self, func_args, func_kwargs, spinner_message)\u001b[39m\n\u001b[32m    263\u001b[39m spinner_or_no_context = (\n\u001b[32m    264\u001b[39m     spinner(spinner_message, _cache=\u001b[38;5;28;01mTrue\u001b[39;00m, show_time=\u001b[38;5;28mself\u001b[39m._info.show_time)\n\u001b[32m    265\u001b[39m     \u001b[38;5;28;01mif\u001b[39;00m spinner_message \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m is_nested_cache_function\n\u001b[32m    266\u001b[39m     \u001b[38;5;28;01melse\u001b[39;00m contextlib.nullcontext()\n\u001b[32m    267\u001b[39m )\n\u001b[32m    268\u001b[39m \u001b[38;5;28;01mwith\u001b[39;00m spinner_or_no_context:\n\u001b[32m--> \u001b[39m\u001b[32m269\u001b[39m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[43m.\u001b[49m\u001b[43m_handle_cache_miss\u001b[49m\u001b[43m(\u001b[49m\u001b[43mcache\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mvalue_key\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mfunc_args\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mfunc_kwargs\u001b[49m\u001b[43m)\u001b[49m\n",
      "\u001b[36mFile \u001b[39m\u001b[32m~\\miniforge3\\Lib\\site-packages\\streamlit\\runtime\\caching\\cache_utils.py:328\u001b[39m, in \u001b[36mCachedFunc._handle_cache_miss\u001b[39m\u001b[34m(self, cache, value_key, func_args, func_kwargs)\u001b[39m\n\u001b[32m    324\u001b[39m \u001b[38;5;66;03m# We acquired the lock before any other thread. Compute the value!\u001b[39;00m\n\u001b[32m    325\u001b[39m \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28mself\u001b[39m._info.cached_message_replay_ctx.calling_cached_function(\n\u001b[32m    326\u001b[39m     \u001b[38;5;28mself\u001b[39m._info.func\n\u001b[32m    327\u001b[39m ):\n\u001b[32m--> \u001b[39m\u001b[32m328\u001b[39m     computed_value = \u001b[38;5;28;43mself\u001b[39;49m\u001b[43m.\u001b[49m\u001b[43m_info\u001b[49m\u001b[43m.\u001b[49m\u001b[43mfunc\u001b[49m\u001b[43m(\u001b[49m\u001b[43m*\u001b[49m\u001b[43mfunc_args\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43m*\u001b[49m\u001b[43m*\u001b[49m\u001b[43mfunc_kwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[32m    330\u001b[39m \u001b[38;5;66;03m# We've computed our value, and now we need to write it back to the cache\u001b[39;00m\n\u001b[32m    331\u001b[39m \u001b[38;5;66;03m# along with any \"replay messages\" that were generated during value computation.\u001b[39;00m\n\u001b[32m    332\u001b[39m messages = \u001b[38;5;28mself\u001b[39m._info.cached_message_replay_ctx._most_recent_messages\n",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[1]\u001b[39m\u001b[32m, line 9\u001b[39m, in \u001b[36mload_model\u001b[39m\u001b[34m()\u001b[39m\n\u001b[32m      6\u001b[39m \u001b[38;5;129m@st\u001b[39m.cache_resource\n\u001b[32m      7\u001b[39m \u001b[38;5;28;01mdef\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34mload_model\u001b[39m():\n\u001b[32m      8\u001b[39m     tfidf = joblib.load(\u001b[33mr\u001b[39m\u001b[33m\"\u001b[39m\u001b[33mC:\u001b[39m\u001b[33m\\\u001b[39m\u001b[33mUsers\u001b[39m\u001b[33m\\\u001b[39m\u001b[33mdwiky\u001b[39m\u001b[33m\\\u001b[39m\u001b[33mDocuments\u001b[39m\u001b[33m\\\u001b[39m\u001b[33mgooglemaps_projectt\u001b[39m\u001b[33m\\\u001b[39m\u001b[33mModels\u001b[39m\u001b[33m\\\u001b[39m\u001b[33mtfidf_vectorizer.pkl\u001b[39m\u001b[33m\"\u001b[39m)\n\u001b[32m----> \u001b[39m\u001b[32m9\u001b[39m     model = \u001b[43mjoblib\u001b[49m\u001b[43m.\u001b[49m\u001b[43mload\u001b[49m\u001b[43m(\u001b[49m\u001b[33;43mr\u001b[39;49m\u001b[33;43m\"\u001b[39;49m\u001b[33;43mC:\u001b[39;49m\u001b[33;43m\\\u001b[39;49m\u001b[33;43mUsers\u001b[39;49m\u001b[33;43m\\\u001b[39;49m\u001b[33;43mdwiky\u001b[39;49m\u001b[33;43m\\\u001b[39;49m\u001b[33;43mDocuments\u001b[39;49m\u001b[33;43m\\\u001b[39;49m\u001b[33;43mgooglemaps_projectt\u001b[39;49m\u001b[33;43m\\\u001b[39;49m\u001b[33;43mModels\u001b[39;49m\u001b[33;43m\\\u001b[39;49m\u001b[33;43msentiment_model.pkl\u001b[39;49m\u001b[33;43m\"\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[32m     10\u001b[39m     \u001b[38;5;28;01mreturn\u001b[39;00m tfidf, model\n",
      "\u001b[36mFile \u001b[39m\u001b[32m~\\miniforge3\\Lib\\site-packages\\joblib\\numpy_pickle.py:735\u001b[39m, in \u001b[36mload\u001b[39m\u001b[34m(filename, mmap_mode, ensure_native_byte_order)\u001b[39m\n\u001b[32m    733\u001b[39m         obj = _unpickle(fobj, ensure_native_byte_order=ensure_native_byte_order)\n\u001b[32m    734\u001b[39m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[32m--> \u001b[39m\u001b[32m735\u001b[39m     \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28;43mopen\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mfilename\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[33;43m\"\u001b[39;49m\u001b[33;43mrb\u001b[39;49m\u001b[33;43m\"\u001b[39;49m\u001b[43m)\u001b[49m \u001b[38;5;28;01mas\u001b[39;00m f:\n\u001b[32m    736\u001b[39m         \u001b[38;5;28;01mwith\u001b[39;00m _validate_fileobject_and_memmap(f, filename, mmap_mode) \u001b[38;5;28;01mas\u001b[39;00m (\n\u001b[32m    737\u001b[39m             fobj,\n\u001b[32m    738\u001b[39m             validated_mmap_mode,\n\u001b[32m    739\u001b[39m         ):\n\u001b[32m    740\u001b[39m             \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(fobj, \u001b[38;5;28mstr\u001b[39m):\n\u001b[32m    741\u001b[39m                 \u001b[38;5;66;03m# if the returned file object is a string, this means we\u001b[39;00m\n\u001b[32m    742\u001b[39m                 \u001b[38;5;66;03m# try to load a pickle file generated with an version of\u001b[39;00m\n\u001b[32m    743\u001b[39m                 \u001b[38;5;66;03m# Joblib so we load it with joblib compatibility function.\u001b[39;00m\n",
      "\u001b[31mFileNotFoundError\u001b[39m: [Errno 2] No such file or directory: 'C:\\\\Users\\\\dwiky\\\\Documents\\\\googlemaps_projectt\\\\Models\\\\sentiment_model.pkl'"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# === Load model dan vectorizer ===\n",
    "@st.cache_resource\n",
    "def load_models():\n",
    "    tfidf = joblib.load(\"Models/tfidf_vectorizer.pkl\")\n",
    "    svm_model = joblib.load(\"Models/svm_sentiment_model.pkl\")\n",
    "    kmeans_model = joblib.load(\"Models/kmeans.pkl\")\n",
    "    return tfidf, svm_model, kmeans_model\n",
    "\n",
    "tfidf, svm_model, kmeans_model = load_models()\n",
    "\n",
    "# === Judul Aplikasi ===\n",
    "st.title(\"📊 Sistem Analisis & Rekomendasi Ulasan Perpustakaan Nasional\")\n",
    "st.markdown(\"Model: **SVM (Sentimen)** + **K-Means (Clustering)** + **TF-IDF (Feature Extraction)**\")\n",
    "\n",
    "# === Input pengguna ===\n",
    "user_input = st.text_area(\"Masukkan ulasan pengguna di sini:\", \"\")\n",
    "\n",
    "if st.button(\"Analisis Ulasan\"):\n",
    "    if user_input.strip():\n",
    "        X = tfidf.transform([user_input])\n",
    "        sentiment_pred = svm_model.predict(X)[0]\n",
    "        cluster_pred = kmeans_model.predict(X)[0]\n",
    "\n",
    "        st.subheader(\"🔍 Hasil Analisis:\")\n",
    "        st.write(f\"**Sentimen:** {sentiment_pred}\")\n",
    "        st.write(f\"**Cluster:** {cluster_pred}\")\n",
    "\n",
    "        # Rekomendasi berdasarkan cluster\n",
    "        if cluster_pred == 0:\n",
    "            st.success(\"📚 Rekomendasi: Ulasan ini mirip dengan kelompok pembaca yang memberikan **ulasan positif** dan sering merekomendasikan layanan pustaka digital.\")\n",
    "        elif cluster_pred == 1:\n",
    "            st.info(\"🧐 Rekomendasi: Ulasan ini termasuk kelompok dengan **penilaian netral**, mungkin perlu peningkatan pelayanan.\")\n",
    "        else:\n",
    "            st.warning(\"😔 Rekomendasi: Termasuk cluster dengan **ulasan negatif**, fokuskan perbaikan pada pengalaman pengunjung.\")\n",
    "\n",
    "    else:\n",
    "        st.warning(\"Harap masukkan teks terlebih dahulu!\")\n",
    "\n",
    "st.markdown(\"---\")\n",
    "st.caption(\"Dibuat oleh Nanda | Analisis Sentimen & Sistem Rekomendasi Perpustakaan 📚\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
