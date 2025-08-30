#!/usr/bin/env python3
"""
Interaktivní CLI pro DeepResearchTool
Poskytuje uživatelsky přívětivé rozhraní pro research dotazy
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Přidání src do path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import ModernResearchAgent


class InteractiveCLI:
    """Interaktivní CLI pro research dotazy"""
    
    def __init__(self):
        self.agent: Optional[ModernResearchAgent] = None
        self.current_profile = "thorough"
        self.history = []
        
    def print_welcome(self):
        """Uvítací zpráva"""
        print("🔍 " + "=" * 60)
        print("   DeepResearchTool - Interaktivní CLI")
        print("   Advanced AI Research Agent")
        print("=" * 63)
        print()
        print("📋 Dostupné příkazy:")
        print("   research <dotaz>    - Spustit research dotaz")
        print("   profile <název>     - Změnit profil (quick/thorough/academic)")
        print("   history            - Zobrazit historii dotazů")
        print("   clear              - Vymazat historii")
        print("   status             - Zobrazit stav agenta")
        print("   help               - Zobrazit nápovědu")
        print("   quit / exit        - Ukončit CLI")
        print()
        print(f"🎯 Aktuální profil: {self.current_profile}")
        print("📝 Tip: Začněte příkazem 'research <váš dotaz>'")
        print()
    
    def print_help(self):
        """Nápověda"""
        print("\n📚 Detailní nápověda:")
        print()
        print("🔍 RESEARCH PŘÍKAZY:")
        print("   research <dotaz>           - Základní research")
        print("   research --audit <dotaz>   - Research s audit módem")
        print("   research --save <dotaz>    - Uložit výsledky do souboru")
        print()
        print("⚙️  KONFIGURACE:")
        print("   profile quick             - Rychlé výsledky (~1 min)")
        print("   profile thorough          - Detailní analýza (~3-5 min)")
        print("   profile academic          - Akademický research (~5-10 min)")
        print()
        print("📊 PROFILY:")
        print("   quick     - Rychlé výsledky, základní citace")
        print("   thorough  - Detailní analýza s plnou verifikací")
        print("   academic  - Akademický standard s peer-review")
        print()
        print("💡 PŘÍKLADY:")
        print("   research What are the latest developments in quantum computing?")
        print("   research --audit Climate change impact on Arctic ecosystems")
        print("   profile quick")
        print("   research AI bias in hiring algorithms")
        print()
    
    async def initialize_agent(self):
        """Inicializace research agenta"""
        try:
            print(f"🚀 Inicializuji research agent (profil: {self.current_profile})...")
            self.agent = ModernResearchAgent(
                profile=self.current_profile,
                use_langgraph=True
            )
            print("✅ Agent inicializován úspěšně!")
            return True
        except Exception as e:
            print(f"❌ Chyba při inicializaci agenta: {e}")
            print("💡 Tip: Zkontrolujte, zda jsou spuštěny potřebné služby (Ollama, Qdrant)")
            return False
    
    async def handle_research(self, query: str, audit: bool = False, save: bool = False):
        """Zpracování research dotazu"""
        if not self.agent:
            if not await self.initialize_agent():
                return
        
        try:
            print(f"\n🔍 Spouštím research pro: '{query}'")
            print(f"📊 Profil: {self.current_profile}")
            print(f"🔧 Audit mód: {'Zapnutý' if audit else 'Vypnutý'}")
            print("⏳ Probíhá research...")
            print("-" * 50)
            
            start_time = time.time()
            result = await self.agent.research(query, audit_mode=audit)
            
            # Přidání do historie
            self.history.append({
                "query": query,
                "timestamp": time.time(),
                "profile": self.current_profile,
                "processing_time": result.get("processing_time", 0)
            })
            
            # Výpis výsledků
            self.print_research_results(result)
            
            # Uložení výsledků
            if save or audit:
                filename = f"research_result_{int(time.time())}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"💾 Výsledky uloženy do: {filename}")
                
        except KeyboardInterrupt:
            print("\n⏹️  Research přerušen uživatelem")
        except Exception as e:
            print(f"\n❌ Chyba během research: {e}")
    
    def print_research_results(self, result: Dict[str, Any]):
        """Výpis výsledků research"""
        print("\n🎉 VÝSLEDKY RESEARCH")
        print("=" * 50)
        
        # Základní informace
        print(f"📝 Dotaz: {result.get('query', 'N/A')}")
        print(f"🏗️  Architektura: {result.get('architecture', 'N/A')}")
        print(f"⏱️  Čas zpracování: {result.get('processing_time', 0):.2f}s")
        print(f"📊 Profil: {result.get('profile', 'N/A')}")
        
        # Plán research (pokud existuje)
        if result.get('plan'):
            print(f"\n📋 Plán výzkumu:")
            for i, step in enumerate(result['plan'], 1):
                print(f"   {i}. {step}")
        
        # Statistiky
        claims = result.get('claims', [])
        citations = result.get('citations', [])
        print(f"\n📈 Statistiky:")
        print(f"   • Nalezené dokumenty: {len(result.get('retrieved_docs', []))}")
        print(f"   • Extrahované claims: {len(claims)}")
        print(f"   • Citace: {len(citations)}")
        
        # Validační skóre
        if result.get('validation_scores'):
            print(f"\n✅ Validační skóre:")
            for metric, score in result['validation_scores'].items():
                print(f"   • {metric}: {score:.2f}")
        
        # Hlavní syntéza
        if result.get('synthesis'):
            print(f"\n📄 SYNTÉZA:")
            print("-" * 30)
            print(result['synthesis'])
        
        # Claims s citacemi
        if claims:
            print(f"\n🎯 KLÍČOVÉ POZNATKY:")
            print("-" * 30)
            for i, claim in enumerate(claims[:5], 1):  # Zobrazit max 5 claims
                confidence = claim.get('confidence', 0)
                confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                print(f"{i}. {confidence_emoji} {claim.get('claim', 'N/A')}")
                print(f"   📊 Confidence: {confidence:.2f}")
        
        # Chyby
        if result.get('errors'):
            print(f"\n⚠️  VAROVÁNÍ:")
            for error in result['errors']:
                print(f"   • {error}")
        
        print("\n" + "=" * 50 + "\n")
    
    def handle_profile(self, profile: str):
        """Změna profilu"""
        valid_profiles = ["quick", "thorough", "academic"]
        if profile in valid_profiles:
            old_profile = self.current_profile
            self.current_profile = profile
            self.agent = None  # Resetovat agenta pro nový profil
            print(f"✅ Profil změněn z '{old_profile}' na '{profile}'")
            print(f"ℹ️  Agent bude reinicializován při příštím dotazu")
        else:
            print(f"❌ Neplatný profil: {profile}")
            print(f"📋 Dostupné profily: {', '.join(valid_profiles)}")
    
    def show_history(self):
        """Zobrazení historie dotazů"""
        if not self.history:
            print("📋 Historie je prázdná")
            return
        
        print(f"\n📚 HISTORIE DOTAZŮ ({len(self.history)} položek):")
        print("-" * 50)
        
        for i, item in enumerate(self.history[-10:], 1):  # Posledních 10
            timestamp = time.strftime("%H:%M:%S", time.localtime(item['timestamp']))
            print(f"{i:2d}. [{timestamp}] [{item['profile']}] {item['query'][:50]}...")
            print(f"     ⏱️  {item['processing_time']:.1f}s")
        
        if len(self.history) > 10:
            print(f"\n... a {len(self.history) - 10} dalších položek")
        print()
    
    def show_status(self):
        """Zobrazení stavu agenta"""
        print(f"\n📊 STAV AGENTA:")
        print("-" * 30)
        print(f"🎯 Profil: {self.current_profile}")
        print(f"🤖 Agent: {'Inicializován' if self.agent else 'Neinicializován'}")
        print(f"📚 Historie: {len(self.history)} dotazů")
        
        if self.agent:
            print(f"🏗️  Architektura: LangGraph")
        
        print()
    
    def clear_history(self):
        """Vymazání historie"""
        count = len(self.history)
        self.history.clear()
        print(f"🗑️  Vymazáno {count} položek z historie")
    
    def parse_command(self, user_input: str) -> tuple:
        """Parsování uživatelského příkazu"""
        parts = user_input.strip().split()
        if not parts:
            return None, []
        
        command = parts[0].lower()
        args = parts[1:]
        
        return command, args
    
    async def run(self):
        """Hlavní smyčka CLI"""
        self.print_welcome()
        
        while True:
            try:
                user_input = input("🔍 DeepResearch> ").strip()
                
                if not user_input:
                    continue
                
                command, args = self.parse_command(user_input)
                
                if command in ['quit', 'exit']:
                    print("👋 Ukončuji CLI. Na shledanou!")
                    break
                
                elif command == 'help':
                    self.print_help()
                
                elif command == 'research':
                    if not args:
                        print("❌ Chybí research dotaz. Použití: research <dotaz>")
                        continue
                    
                    # Zpracování přepínačů
                    audit = '--audit' in args
                    save = '--save' in args
                    
                    # Odstranění přepínačů z dotazu
                    query_parts = [arg for arg in args if not arg.startswith('--')]
                    query = ' '.join(query_parts)
                    
                    if not query:
                        print("❌ Chybí research dotaz")
                        continue
                    
                    await self.handle_research(query, audit=audit, save=save)
                
                elif command == 'profile':
                    if not args:
                        print(f"📊 Aktuální profil: {self.current_profile}")
                        print("📋 Dostupné profily: quick, thorough, academic")
                    else:
                        self.handle_profile(args[0])
                
                elif command == 'history':
                    self.show_history()
                
                elif command == 'status':
                    self.show_status()
                
                elif command == 'clear':
                    self.clear_history()
                
                else:
                    print(f"❌ Neznámý příkaz: {command}")
                    print("💡 Napište 'help' pro nápovědu")
            
            except KeyboardInterrupt:
                print("\n👋 Ukončuji CLI. Na shledanou!")
                break
            except EOFError:
                print("\n👋 Ukončuji CLI. Na shledanou!")
                break
            except Exception as e:
                print(f"❌ Neočekávaná chyba: {e}")


async def main():
    """Hlavní funkce CLI"""
    cli = InteractiveCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())