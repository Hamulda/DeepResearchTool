# Krok 1: Použijeme oficiální odlehčený Python image jako základ
FROM python:3.11-slim

# Krok 2: Nastavíme pracovní adresář uvnitř kontejneru
WORKDIR /app

# Krok 3: Zkopírujeme soubor se závislostmi a nainstalujeme je
# (Děláme to odděleně, aby Docker mohl využít cache, pokud se změní jen kód)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Krok 4: Zkopírujeme zbytek aplikace do kontejneru
COPY . .

# Krok 5: Definujeme výchozí příkaz, který se spustí při startu kontejneru
CMD ["python", "main.py"]