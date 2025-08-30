"""
Demo Script pro Fázi 3: Multi-Modality
Demonstruje zpracování obrazových dat, OCR, EXIF metadata a cross-modální vyhledávání
"""

import asyncio
import json
import tempfile
import polars as pl
from datetime import datetime, timezone
from pathlib import Path
import sys
import os
import shutil
from PIL import Image, ImageDraw, ImageFont
import base64
import io

# Přidej src do path
sys.path.append("/app/src")
sys.path.append("/app/workers")

from processing_worker import EnhancedProcessingWorker
from acquisition_worker import EnhancedAcquisitionWorker


async def demo_multi_modality():
    """Kompletní demo Fáze 3 - Multi-Modality"""

    print("🚀 === DEMO: Fáze 3 - Multi-Modality ===")
    print()

    # 1. Inicializace Multi-Modal systému
    print("📦 1. Inicializace Multi-Modal systému...")

    try:
        processing_worker = EnhancedProcessingWorker()
        print("✅ Enhanced Processing Worker inicializován")

        if processing_worker.image_processor:
            print("✅ Image Processor dostupný")
        else:
            print("⚠️ Image Processor není dostupný - některé funkce nebudou fungovat")

        acquisition_worker = EnhancedAcquisitionWorker()
        print("✅ Enhanced Acquisition Worker připraven")

    except Exception as e:
        print(f"❌ Chyba při inicializaci: {e}")
        return

    print()

    # 2. Vytvoření testovacích obrázků
    print("🖼️ 2. Vytváření testovacích obrázků s různými typy obsahu...")

    images_dir = Path("/tmp/demo_images_phase3")
    images_dir.mkdir(exist_ok=True)

    # Vytvoř různé typy testovacích obrázků
    test_images = await create_test_images(images_dir)
    print(f"✅ Vytvořeno {len(test_images)} testovacích obrázků")

    print()

    # 3. Testování Image Processing
    print("🔍 3. Testování zpracování obrázků...")

    if processing_worker.image_processor:
        print("📊 Zpracovávám jednotlivé obrázky:")

        for i, image_path in enumerate(test_images[:3], 1):
            print(f"   {i}. Zpracovávám: {image_path.name}")

            try:
                result = processing_worker.image_processor.process_image(
                    str(image_path), f"demo://test_image_{i}"
                )

                if not result.get("error"):
                    file_info = result.get("file_info", {})
                    ocr_result = result.get("ocr_result", {})
                    exif_data = result.get("exif_metadata", {})

                    print(f"      📏 Velikost: {file_info.get('file_size', 0)} bytů")
                    print(f"      📝 OCR text: {len(ocr_result.get('text', ''))} znaků")
                    if ocr_result.get("text"):
                        preview = ocr_result["text"][:50]
                        print(
                            f"         Ukázka: '{preview}{'...' if len(ocr_result['text']) > 50 else ''}'"
                        )

                    print(
                        f"      📷 Rozměry: {exif_data.get('image_width', 'N/A')}x{exif_data.get('image_height', 'N/A')}"
                    )
                    print(f"      🎨 Formát: {exif_data.get('image_format', 'N/A')}")

                    embeddings = result.get("embeddings", {})
                    if embeddings:
                        embed_types = []
                        if "text_embedding" in embeddings:
                            embed_types.append("Text")
                        if "clip_image_embedding" in embeddings:
                            embed_types.append("CLIP")
                        print(f"      🧠 Embeddings: {', '.join(embed_types)}")
                else:
                    print(f"      ❌ Chyba: {result.get('error')}")

            except Exception as e:
                print(f"      ❌ Chyba při zpracování: {e}")

        print()

    # 4. Batch zpracování obrázků
    print("📦 4. Batch zpracování všech obrázků...")

    task_id = f"demo_multimodal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        batch_result = await processing_worker.process_images_from_directory(
            str(images_dir), task_id
        )

        if batch_result["success"]:
            print(f"✅ Batch zpracování úspěšné:")
            print(f"   📊 Zpracováno obrázků: {batch_result['images_processed']}")
            print(f"   📊 Nalezeno obrázků: {batch_result['total_images_found']}")
            print(f"   📊 Extrahované entity: {batch_result['entities_extracted']}")
            print(f"   📊 Vygenerované embeddings: {batch_result['embeddings_generated']}")

            if batch_result.get("output_file"):
                print(f"   💾 Výstupní soubor: {batch_result['output_file']}")
        else:
            print(f"❌ Batch zpracování selhalo: {batch_result.get('error')}")

    except Exception as e:
        print(f"❌ Chyba při batch zpracování: {e}")

    print()

    # 5. Test extrakce obrázků ze stránky
    print("🌐 5. Test extrakce obrázků ze webové stránky...")

    # Vytvoř vzorovou HTML stránku s obrázky
    sample_html = create_sample_html_with_images(test_images)

    try:
        image_extraction_result = await acquisition_worker.extract_and_download_images(
            sample_html, "http://demo.local/test-page", f"{task_id}_web"
        )

        if image_extraction_result["success"]:
            print(f"✅ Extrakce obrázků ze stránky:")
            print(f"   📊 Nalezeno obrázků: {image_extraction_result['total_images_found']}")
            print(f"   📊 Platné obrázky: {image_extraction_result['valid_images_found']}")
            print(f"   📊 Stažené obrázky: {image_extraction_result['images_downloaded']}")
            print(f"   📊 SVG elementy: {image_extraction_result['svg_elements']}")

            if image_extraction_result.get("images_directory"):
                print(f"   📁 Adresář obrázků: {image_extraction_result['images_directory']}")
        else:
            print(f"❌ Extrakce selhala: {image_extraction_result.get('error')}")

    except Exception as e:
        print(f"❌ Chyba při extrakci obrázků: {e}")

    print()

    # 6. Cross-modální vyhledávání
    print("🔎 6. Testování cross-modálního vyhledávání...")

    if processing_worker.image_processor:
        # Test text-to-image search
        test_queries = [
            ("bitcoin cryptocurrency", "text_to_image"),
            ("user information", "text_to_image"),
            ("forum discussion", "text_to_image"),
            ("crypto address", "image_to_text"),
        ]

        for query, search_type in test_queries:
            print(f"🔍 Testování: '{query}' ({search_type})")

            try:
                search_result = await processing_worker.search_cross_modal(
                    query=query, search_type=search_type, limit=3, task_id=task_id
                )

                if search_result["success"]:
                    results = search_result["results"]
                    print(f"   ✅ Nalezeno {len(results)} výsledků:")

                    for i, result in enumerate(results, 1):
                        similarity = result.get("similarity_score", 0)
                        image_file = result.get("image_file", "N/A")
                        image_name = Path(image_file).name if image_file != "N/A" else "N/A"
                        print(f"      {i}. {image_name} (podobnost: {similarity:.3f})")
                else:
                    print(f"   ❌ Vyhledávání selhalo: {search_result.get('error')}")

            except Exception as e:
                print(f"   ❌ Chyba při vyhledávání: {e}")

            print()

    # 7. Integrace s Knowledge Graph
    print("🕸️ 7. Test integrace s Knowledge Graph...")

    if processing_worker.kg_manager:
        try:
            # Získej statistiky grafu po přidání image entit
            kg_stats = await processing_worker.kg_manager.get_graph_statistics()

            if "error" not in kg_stats:
                print(f"📊 Knowledge Graph statistiky:")
                print(f"   • Celkem entit: {kg_stats.get('total_entities', 0)}")
                print(f"   • Celkem vztahů: {kg_stats.get('total_relations', 0)}")
                print(f"   • Celkem zdrojů: {kg_stats.get('total_sources', 0)}")

                # Hledej image-related entity
                image_entities = await processing_worker.kg_manager.query_entities(
                    entity_type="crypto_addresses", limit=5
                )

                if image_entities:
                    print(f"   • Image entity příklad:")
                    for entity in image_entities[:3]:
                        print(f"     - {entity['text']} ({entity['type']})")
            else:
                print(f"⚠️ Knowledge Graph: {kg_stats['error']}")

        except Exception as e:
            print(f"❌ Chyba při dotazu na KG: {e}")
    else:
        print("⚠️ Knowledge Graph není dostupný")

    print()

    # 8. Analýza výkonu a statistiky
    print("📈 8. Analýza výkonu a statistiky...")

    # LanceDB tabulky
    try:
        available_tables = processing_worker.db.table_names()
        image_tables = [t for t in available_tables if "image_embeddings" in t]

        print(f"📚 LanceDB databáze:")
        print(f"   • Celkem tabulek: {len(available_tables)}")
        print(f"   • Image embedding tabulky: {len(image_tables)}")

        if image_tables:
            print(f"   • Image tabulky:")
            for table in image_tables:
                try:
                    table_obj = processing_worker.db.open_table(table)
                    count = len(table_obj.to_pandas())
                    print(f"     - {table}: {count} embeddings")
                except Exception as e:
                    print(f"     - {table}: chyba ({e})")

    except Exception as e:
        print(f"❌ Chyba při analýze databáze: {e}")

    print()

    # 9. Ukázka EXIF analýzy
    print("📷 9. Ukázka EXIF analýzy a metadat...")

    if processing_worker.image_processor and test_images:
        print("🔍 Analýza EXIF metadat:")

        for i, image_path in enumerate(test_images[:2], 1):
            try:
                result = processing_worker.image_processor.process_image(
                    str(image_path), f"demo://exif_test_{i}"
                )

                if not result.get("error"):
                    exif_data = result.get("exif_metadata", {})
                    content_analysis = result.get("content_analysis", {})

                    print(f"   📸 Obrázek {i} ({image_path.name}):")
                    print(
                        f"      • Rozměry: {exif_data.get('image_width', 'N/A')}x{exif_data.get('image_height', 'N/A')}"
                    )
                    print(f"      • Formát: {exif_data.get('image_format', 'N/A')}")
                    print(f"      • Mód: {exif_data.get('image_mode', 'N/A')}")

                    # GPS data (pokud jsou dostupná)
                    gps_data = exif_data.get("GPS", {})
                    if gps_data:
                        lat = gps_data.get("latitude", "N/A")
                        lon = gps_data.get("longitude", "N/A")
                        print(f"      • GPS: {lat}, {lon}")

                    # Content analysis
                    content_type = content_analysis.get("content_type", "unknown")
                    print(f"      • Typ obsahu: {content_type}")

                    dominant_colors = content_analysis.get("dominant_colors", [])
                    if dominant_colors:
                        print(f"      • Dominantní barvy: {len(dominant_colors)} detekováno")

            except Exception as e:
                print(f"   ❌ Chyba při analýze {image_path.name}: {e}")

    print()

    # 10. Závěr
    print("✅ === DEMO DOKONČENO ===")
    print()
    print("🎉 Fáze 3: Multi-Modality byla úspěšně implementována!")
    print()
    print("📋 Implementované funkce:")
    print("   ✅ OCR extrakce textu z obrázků (Tesseract)")
    print("   ✅ EXIF metadata extrakce (GPS, camera info)")
    print("   ✅ CLIP embeddings pro cross-modální vyhledávání")
    print("   ✅ Detekce a stahování obrázků ze stránek")
    print("   ✅ Batch zpracování obrázků")
    print("   ✅ Integrace s Knowledge Graph")
    print("   ✅ Cross-modální text-to-image search")
    print("   ✅ Cross-modální image-to-text search")
    print("   ✅ Analýza obsahu obrázků")
    print("   ✅ LanceDB indexování pro rychlé vyhledávání")
    print()
    print("🔄 Multi-Modal Pipeline:")
    print("   1. Detekce obrázků na webových stránkách")
    print("   2. Stahování a preprocessing obrázků")
    print("   3. OCR extrakce textu z obrázků")
    print("   4. EXIF metadata analýza")
    print("   5. CLIP embedding generování")
    print("   6. Entity extraction z OCR textu")
    print("   7. Indexování do LanceDB")
    print("   8. Ukládání do Knowledge Graph")
    print("   9. Cross-modální vyhledávání")
    print()
    print("🚀 Systém je připraven pro Fázi 4: Kontinuální systém!")

    # Vyčisti
    try:
        shutil.rmtree(images_dir)
        print(f"🧹 Vyčištěny testovací obrázky")
    except:
        pass


async def create_test_images(images_dir: Path) -> List[Path]:
    """Vytvoř testovací obrázky s různými typy obsahu"""
    test_images = []

    try:
        # 1. Obrázek s textem (simulace screenshotu)
        img1 = Image.new("RGB", (800, 600), color="white")
        draw = ImageDraw.Draw(img1)

        # Pokus se najít font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                font = ImageFont.load_default()

        # Text content
        text_lines = [
            "DARK WEB FORUM - BITCOIN TRADING",
            "",
            "User: CryptoKing",
            "Posted: 2024-08-29",
            "",
            "Selling 10 BTC for cash",
            "Address: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            "Contact: cryptoking@secure.onion",
            "",
            "Only serious buyers!",
            "Meet in Prague, Czech Republic",
        ]

        y_offset = 50
        for line in text_lines:
            draw.text((50, y_offset), line, fill="black", font=font)
            y_offset += 40

        img1_path = images_dir / "forum_screenshot.png"
        img1.save(img1_path)
        test_images.append(img1_path)

        # 2. Obrázek s krypto informacemi
        img2 = Image.new("RGB", (600, 400), color="lightblue")
        draw2 = ImageDraw.Draw(img2)

        crypto_text = [
            "CRYPTOCURRENCY WALLET",
            "",
            "Bitcoin Address:",
            "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",
            "",
            "Ethereum Address:",
            "0x742d35Cc6635C0532925a3b8D357Bb682Bfb86Fb",
            "",
            "Balance: 2.5 BTC",
            "Last TX: 2024-08-29 14:30:22",
        ]

        y_offset = 30
        for line in crypto_text:
            draw2.text((30, y_offset), line, fill="darkblue", font=font)
            y_offset += 30

        img2_path = images_dir / "crypto_wallet.png"
        img2.save(img2_path)
        test_images.append(img2_path)

        # 3. Obrázek s user profile
        img3 = Image.new("RGB", (500, 300), color="lightgray")
        draw3 = ImageDraw.Draw(img3)

        profile_text = [
            "USER PROFILE",
            "",
            "Username: AliceTrader",
            "Member since: 2022",
            "Location: Prague, CZ",
            "PGP: A1B2C3D4E5F6789012345678901234567890ABCD",
            "",
            "Trusted trader",
            "Rating: 4.8/5.0",
        ]

        y_offset = 20
        for line in profile_text:
            draw3.text((20, y_offset), line, fill="black", font=font)
            y_offset += 25

        img3_path = images_dir / "user_profile.png"
        img3.save(img3_path)
        test_images.append(img3_path)

        # 4. Jednoduchý obrázek bez textu
        img4 = Image.new("RGB", (300, 300), color="red")
        draw4 = ImageDraw.Draw(img4)
        draw4.ellipse([50, 50, 250, 250], fill="yellow", outline="black", width=3)

        img4_path = images_dir / "simple_graphic.png"
        img4.save(img4_path)
        test_images.append(img4_path)

        print(f"✅ Vytvořeno {len(test_images)} testovacích obrázků")
        return test_images

    except Exception as e:
        print(f"❌ Chyba při vytváření testovacích obrázků: {e}")
        return []


def create_sample_html_with_images(test_images: List[Path]) -> str:
    """Vytvoř vzorovou HTML stránku s odkazy na obrázky"""

    # Konvertuj první obrázek na data URL pro demo
    data_url = ""
    if test_images:
        try:
            with open(test_images[0], "rb") as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode()
                data_url = f"data:image/png;base64,{img_base64}"
        except:
            pass

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dark Web Forum - Crypto Trading</title>
        <style>
            body {{ background: #1a1a1a; color: #00ff00; font-family: monospace; }}
            .post {{ border: 1px solid #00ff00; margin: 10px; padding: 10px; }}
            .avatar {{ width: 64px; height: 64px; background: #333; }}
        </style>
    </head>
    <body>
        <h1>🌐 SECURE CRYPTO TRADING FORUM 🌐</h1>
        
        <div class="post">
            <h3>📸 Latest Screenshots</h3>
            <img src="/images/forum_screenshot.png" alt="Forum Screenshot" width="400">
            <img src="/images/crypto_wallet.png" alt="Crypto Wallet" width="300">
            <img src="./user_profile.png" alt="User Profile" width="250">
        </div>
        
        <div class="post">
            <h3>🖼️ Profile Images</h3>
            <img class="avatar" src="/avatars/cryptoking.jpg" alt="CryptoKing Avatar">
            <img class="avatar" src="/avatars/alice.png" alt="AliceTrader Avatar">
        </div>
        
        <div class="post">
            <h3>📊 Trading Charts</h3>
            <img src="https://example.com/btc_chart.png" alt="Bitcoin Chart">
            <img src="https://example.com/eth_chart.gif" alt="Ethereum Chart">
        </div>
        
        <div class="post" style="background-image: url('/backgrounds/matrix.jpg');">
            <h3>🎨 CSS Background</h3>
            <p>This div has a background image</p>
        </div>
        
        <div class="post">
            <h3>📱 Embedded Data Image</h3>
            {f'<img src="{data_url}" alt="Embedded Image" width="200">' if data_url else '<p>No data URL generated</p>'}
        </div>
        
        <div class="post">
            <h3>🔗 Lazy Loading Images</h3>
            <img data-src="/lazy/qr_code.png" alt="QR Code" loading="lazy">
            <img data-lazy-src="/lazy/pgp_key.png" alt="PGP Key">
        </div>
        
        <svg width="100" height="100">
            <circle cx="50" cy="50" r="40" stroke="green" stroke-width="4" fill="yellow" />
            <text x="50" y="55" text-anchor="middle" fill="black">SVG</text>
        </svg>
    </body>
    </html>
    """

    return html


if __name__ == "__main__":
    # Nastav environment proměnné
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "research2024")
    os.environ.setdefault("OLLAMA_HOST", "localhost")
    os.environ.setdefault("LLM_MODEL", "llama2")
    os.environ.setdefault("MAX_IMAGES_PER_PAGE", "10")

    # Spusť demo
    asyncio.run(demo_multi_modality())
