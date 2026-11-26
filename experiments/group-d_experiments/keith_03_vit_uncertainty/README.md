# Keith 03 · `vit_uncertainty`: Vorhersage und Unsicherheit von Vision Transformern erklären

**Leitfrage des Projekts:** Welche Merkmale machen ein Modell unsicher, und ist diese Unsicherheit Rauschen
in den Daten (aleatorisch, AU) oder fehlendes Wissen (epistemisch, EU)?

Dieser Ordner enthält einen **Prototyp einer shapiq-Erweiterung**: das Paket `vit_uncertainty`. Es nimmt ein
Bildmodell, macht daraus mit **probly** ein unsicherheitsbewusstes Modell und erklärt mit **shapiq** für jedes
Bild die Vorhersage **und** ihre Unsicherheit, getrennt in AU und EU. Datensatz ist Oxford-IIIT Pet mit genau
der Aufteilung aus Dans VGG16-Reihe (Branch `dan/oxford-pets-vgg16`).

## Kurzfassung

- **Das Paket funktioniert.** `UncertaintyExplainer` ist ein echter `shapiq.Explainer`. Er erklärt für jedes Bild
  die Vorhersage (P) und ihre Unsicherheit (TU, AU, EU) eines beliebigen probly-Modells: Ensemble, MC Dropout oder
  ein Ensemble aus zwei verschiedenen Transformern. Effizienz und φ(TU) = φ(AU) + φ(EU) gelten, 40 Tests sind grün,
  und `ruff` und `ty` finden keine Fehler.
- **Zwei verschiedene Netze zeigen mehr fehlendes Wissen.** Das gemischte Ensemble (ViT + DINOv2) hat 2,5-mal so viel
  EU wie Köpfe nur auf dem ViT; 73 % davon entstehen **zwischen** den Netzen. Das bestätigt unseren Vorversuch
  (siehe „Vorversuch“ unten).
- **Tokens weglassen ist ehrlicher als Übermalen.** Ist alles ausgeblendet, sagt das Modell „keine Ahnung“
  (TU 3,45 nats, nahe am Maximum ln 37 ≈ 3,61). Ist alles weichgezeichnet, gibt das Modell der vorhergesagten Rasse
  im Mittel noch 50 %: Der Weichzeichner entfernt die Information gar nicht. Die EU-Karten unterscheiden sich stark
  (ρ = 0,35).
- **Fehlendes Wissen liegt eher auf dem Tier** (+10,5 Prozentpunkte über dem Flächenanteil; Dan mit VGG16 in derselben
  Zählweise: +11,1) und hängt stark an **Kombinationen** von Bildteilen: Bei EU sind die Paar-Beiträge zusammen dreimal
  so groß wie die Einzelbeiträge, bei P nur anderthalbmal.
- **EU erkennt sichere Fehler nicht.** Ein Staffordshire Bull Terrier wird mit 96 % als American Bulldog vorhergesagt,
  bei einer EU von nur 0,010 nats.
- **Ground Truth: Die Paare zeigen auf die richtige Stelle, aber nur der Vergleich mit einer Kontrolle ist
  belastbar.** Auf Testbildern mit eingebauter Ursache (03-3) erhöht eine Chimäre (Kopf von Rasse A auf dem Körper von
  Rasse B) das fehlende Wissen, sobald Kopf und Körper zusammen sichtbar sind. shapiq zeigt das als Paar Kopf × Körper
  (k-SII +0,104 gegen −0,002 bei der Kontrolle). Größer als bei der eigenen Kontrolle ist das Paar mit jedem der vier
  Indizes k-SII, STII, FSII und Möbius (73 bis 77 % der Bilder) und auch mit 16 Superpixeln (85 %). Ob es über null
  liegt, hängt dagegen am Index und an den Spielern (F12 und F13 nicht gestützt). Ein fremdes Schachbrett macht das
  Modell unsicher, aber als **Rauschen**: Alle Mitglieder sagen gemeinsam „keine Ahnung“. EU ist hier also kein Alarm
  für Fremdes.
- **ViT Shapley für Unsicherheit** liefert alle vier Karten in einem Durchlauf, etwa 250-mal schneller als KernelSHAP
  mit 256 Koalitionen. Für TU und AU ist er schon genauer als KernelSHAP 256, für P und EU noch nicht. Die Lernkurve
  zeigt: Mehr Trainings-Koalitionen verbessern P und EU.

## Der Ablauf in fünf Schritten

```
Bild (224 × 224)
  │  1. Modell: zwei eingefrorene Vision Transformer (ViT-S/16, DINOv2-S/14)        backbones.py
  ▼
CLS-Merkmale (384 + 384)
  │  2. Modell → probly: Köpfe, Ensembles, MC Dropout, gemischtes Ensemble           probly_bridge.py
  ▼
probly-Repräsentation ──► quantify ──► TU = AU + EU
  │  3. shapiq-Explainer: Superpixel als Spieler, Tokens ausblenden, k-SII           game.py, explainer.py
  ▼
InteractionValues für P, TU, AU, EU
  │  4. Erklärung von beidem: Vorhersage und Unsicherheit nebeneinander              plot.py
  ▼
  5. ViT Shapley: ein lernender Erklärer liefert alle vier Karten in einem Durchlauf  amortized.py
```

So sieht das für einen Nutzer aus:

```python
from vit_uncertainty import UncertaintyExplainer, load_backbones, plot_prediction_and_uncertainty

backbones = load_backbones()  # 1. Modell
predictor = methods["Gemischt"]  # 2. Modell -> probly (siehe Notebook 03-0)
explainer = UncertaintyExplainer(backbones, predictor)  # 3. ein shapiq.Explainer
explanation = explainer.explain(image)  #    P, TU, AU, EU als InteractionValues
plot_prediction_and_uncertainty(explanation)  # 4. Erklärung von beidem
```

## Dateien

| Datei | Inhalt |
|---|---|
| `vit_uncertainty/data.py` | Oxford-IIIT Pet mit Dans Splits, Bildausschnitt, Kohorten und Kopf-Rahmen |
| `vit_uncertainty/backbones.py` | `FrozenViT`: eingefrorener Transformer, der Patch-Tokens ausblenden kann |
| `vit_uncertainty/players.py` | Spieler: SLIC-Superpixel (wie Tom) oder Raster, Zuordnung zu Tokens |
| `vit_uncertainty/probly_bridge.py` | Köpfe trainieren, probly-Methoden bauen (auch gemischte Ensembles) |
| `vit_uncertainty/removal.py` | `CoalitionFeatures`: Koalitionen einmal durch die Netze schicken, mit Cache |
| `vit_uncertainty/game.py` | `UncertaintyGame`: ein `shapiq.Game` für P, TU, AU oder EU |
| `vit_uncertainty/explainer.py` | `UncertaintyExplainer`: ein `shapiq.Explainer`, liefert `UncertaintyExplanation` |
| `vit_uncertainty/plot.py` | Vorhersage und Unsicherheit nebeneinander, Paare, Vergleiche, Superpixel als Bildchen für shapiqs `si_graph_plot` |
| `vit_uncertainty/amortized.py` | `AmortizedUncertaintyExplainer`: ViT Shapley für Unsicherheit |
| `vit_uncertainty/features.py` | Caches für Merkmale, maskierte Trainingsdaten und Koalitionen |
| `vit_uncertainty/ground_truth.py` | Testbilder mit bekannter Ursache (Muster, zwei Rassen, Chimäre), grobe Spieler, Kennzahlen, feine Paare nach groben Teilen |
| `vit_uncertainty/settings.py` | alle festen Einstellungen an einem Ort |
| `tests/` | Tests mit kleinen Zufalls-Transformern (laufen in Sekunden, ohne Download) |
| `scripts/benchmark.py` | misst die Geschwindigkeit deines Rechners und schätzt alle Laufzeiten |
| `scripts/precompute.py` | rechnet alle Transformer-Läufe vorab (optional, z. B. über Nacht) |
| `scripts/make_pipeline_figure.py` | zeichnet die Ablauf-Abbildung `figures/ablauf.png` |
| `ruff.toml`, `ty.toml` | Einstellungen für Ruff und ty in diesem Ordner (das Repository nimmt `experiments/` von beiden aus) |
| `keith_03-0_vit-probly.ipynb` | Schritt 1 und 2: Daten, Transformer, Köpfe, probly-Methoden, F1 und F2 |
| `keith_03-1_uncertainty-explainer.ipynb` | Schritt 3 und 4: der Explainer, Leitfrage, F3 bis F6 |
| `keith_03-2_vit-shapley.ipynb` | Schritt 5: lernender Erklärer gegen KernelSHAP, F7 |
| `keith_03-3_ground-truth.ipynb` | Ground Truth: Testbilder mit eingebauter Ursache, F8 bis F11 |
| `figures/` | Abbildungen für die Präsentation |

## Ausführen

Vom Hauptordner des Repositorys aus:

```bash
uv sync --group torch
uv pip install shapiq timm scikit-image scikit-learn matplotlib pandas jupyter pytest
uv run python -c "from torchvision.datasets import OxfordIIITPet; OxfordIIITPet('data', download=True)"   # falls Dans Daten fehlen
uv run python experiments/group-d_experiments/keith_03_vit_uncertainty/scripts/benchmark.py              # 1-2 Minuten
uv run pytest experiments/group-d_experiments/keith_03_vit_uncertainty/tests -q                         # ca. 20 Sekunden
```

Danach die Notebooks in der Reihenfolge 03-0, 03-1, 03-2 ausführen; 03-3 (Ground Truth) braucht nur 03-0. Die Gewichte der beiden Transformer
(ca. 180 MB) lädt `timm` beim ersten Lauf von Hugging Face. Alle Zwischenergebnisse liegen in
`artifacts/keith-vit-uncertainty/` (nicht im Repository).

**Laufzeiten** (gemessen auf 2 CPU-Kernen; dein Laptop mit 10 Kernen ist deutlich schneller, siehe `benchmark.py`):

| Schritt | 2 CPU-Kerne (gemessen) | Bemerkung |
|---|---|---|
| Merkmale aller 7349 Bilder (beide Netze) | ca. 15 min | einmalig, danach Cache |
| Teilweise ausgeblendete Trainingsbilder (23.552 + 2944 Koalitionen) | ca. 60 min | einmalig, für maskierte Köpfe und ViT Shapley |
| 24 Köpfe trainieren | ca. 5 bis 11 min | einmalig |
| Koalitionen einer Erklärung (Budget 1000, beide Netze) | ca. 1,8 min je Bild | einmalig je Bild und Spieler-Art; `scripts/precompute.py` macht alles am Stück |
| Erklärung aus dem Cache: Ensemble | ca. 0,2 s je Bild | P, TU, AU, EU mit Shapley-Werten und k-SII |
| Erklärung aus dem Cache: MC Dropout | ca. 17 s je Bild | Bild für Bild mit festen Dropout-Masken |
| Lernenden Erklärer trainieren | wenige Minuten | 40 Epochen |
| Ground Truth (03-3): 180 exakte Erklärungen mit 2 oder 3 Spielern | ca. 2,5 min | 8 Koalitionen je Bild statt 1000 |
| Ground Truth (03-3): dieselben Spiele mit vier Indizes (F12, 480 Erklärungen) | ca. 1 min | keine neuen Transformer-Läufe |
| Ground Truth (03-3): 80 feine Erklärungen (16 SLIC-Superpixel, Budget 1000; je 40 für Muster und Chimäre) | ca. 105 min | einmalig, danach Cache in `artifacts/keith-vit-uncertainty/ground_truth/`; FSII aus denselben Koalitionen kostet nichts |
| Lernender Erklärer: eine Erklärung | unter 0,1 s: ein Lauf durch beide Netze, der Erklärer selbst höchstens etwa 1 ms | statt 1000 Läufen |

Laufen mehrere Rechnungen gleichzeitig, teilen sie sich die Kerne und jede wird entsprechend langsamer.

**Versionen:** `uv sync` allein installiert ältere Pakete (z. B. torch 2.10, scikit-learn 1.8). Geprüft ist alles mit den
Versionen im Protokoll unten; danach `uv run --no-sync` benutzen, damit `uv` sie nicht zurücksetzt.

**Code prüfen** (vor jedem Commit; aus diesem Ordner):

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync ty check          # liest ty.toml; prüft vit_uncertainty/, tests/ und scripts/
```

**Warum eine eigene `ty.toml`?** Das Repository nimmt `experiments/` von ty aus. Ohne eigene Einstellungen würde ty
unseren Code also nie prüfen; so waren bis zum 25.11. 41 Typfehler unbemerkt geblieben. Die probly-Stubs des
Repositorys (`stubs/`, von `prek` erzeugt) lassen wir weg: Sie beschreiben nur `probly.method` und
`probly.transformation`, und Code außerhalb von `src/` findet dann die anderen probly-Module nicht. Ohne sie liest ty
probly direkt aus `src/`.

## Vorversuch (nicht im Repository)

Vor dieser Reihe haben wir dieselbe Frage an Hautbildern untersucht (DermaMNIST aus MedMNIST, 224 × 224 Pixel, 7 Arten
von Hautveränderungen, mit VGG16 und ResNet-50). Drei Erkenntnisse haben diese Reihe geprägt. Erstens sind sich Köpfe
auf einem Netz fast immer einig; erst Köpfe auf verschiedenen Netzen zeigen fehlendes Wissen (3,5- bis 18-mal so viel
EU). Zweitens verändert Übermalen die Erklärungen der Unsicherheit stärker als die der Vorhersage (ρ 0,38 gegen 0,58).
Drittens machen verschiedene Netze verschiedene Bildregionen für EU verantwortlich (ρ 0,07). Der Vorversuch wird nicht
weiterverfolgt.

## Entscheidungen und warum

| Entscheidung | Warum |
|---|---|
| **Oxford-IIIT Pet mit Dans Aufteilung** | Gleiche Bilder, gleicher 224-Pixel-Ausschnitt, gleiche Kohorten wie in Dans VGG16-Reihe. So sind die Zahlen direkt vergleichbar. Dans Code wird nicht importiert; `data.py` baut die Aufteilung nach. |
| **Vision Transformer statt CNN** | Ein ViT verarbeitet das Bild als Tokens. Einen Bildteil kann man **weglassen** (Tokens ausblenden), statt ihn mit einer Farbe zu übermalen. Im Vorversuch hatte gerade das Übermalen die Erklärungen der Unsicherheit stark verändert. |
| **Zwei verschiedene Transformer** (ViT-S/16 überwacht, DINOv2-S/14 selbstüberwacht) | Aus dem Vorversuch: Köpfe auf einem Netz sind sich fast immer einig, ihre EU ist fast null. Mitglieder, die das Bild verschieden „sehen“, zeigen fehlendes Wissen. |
| **Kleine „S“-Modelle, eingefroren** | Für eine Erklärung läuft jedes Bild hunderte Male durch beide Netze. Die S-Modelle sind etwa dreimal so schnell wie ViT-B und laufen auf einer CPU. Nur kleine Köpfe werden trainiert (Dans Einstellungen). |
| **Tokens per Attention-Maske ausblenden** | Für das CLS-Token ist das exakt dasselbe wie die Tokens zu löschen (getestet), lässt sich aber für viele Koalitionen in einem Stapel rechnen. |
| **16 SLIC-Superpixel, exakt 16** | Wie Toms Vergleich (16 Segmente, Kompaktheit 10). SLIC liefert auf 224-Pixel-Bildern meist nur 10 bis 14; wir erhöhen die Anfrage und verschmelzen dann die kleinsten, bis es genau 16 sind. So hat jedes Bild gleich viele Spieler und ein Budget bedeutet überall dasselbe. Jeder Token gehört zu dem Superpixel, das die meisten seiner Pixel bedeckt. |
| **k-SII bis Ordnung 2, Budget 1000** | Wie bei Tom. Einzelbeiträge und Paare; 1000 liegt deutlich über den 137 Unbekannten (1 + 16 + 120). |
| **Vier Spiele (P, TU, AU, EU) teilen sich alles** | Gleicher Seed → shapiq zieht dieselben Koalitionen; ein Cache sorgt dafür, dass jede Koalition nur einmal durch die Netze und nur einmal durch probly geht. |
| **Gemischtes Ensemble über `FeatureSlice`** | `probly.method.ensemble` kopiert ein Grundmodell. Unsere Mitglieder sitzen auf verschiedenen Netzen; jedes liest denselben Vektor und schneidet seinen Teil heraus. |
| **Das Ensemble ist eine `nn.ModuleList`, die wir selbst bauen** | So baut auch probly seine torch-Ensembles, und probly behandelt jede `nn.ModuleList` als Ensemble. Jeder Kopf wird bei probly als „logit classifier“ angemeldet (`as_logit_classifier`). Bis zum 25.11. ließen wir probly Kopien bauen und tauschten sie danach aus; das funktionierte, war aber nicht der vorgesehene Weg. Ein Test zeigt, dass P, TU und AU genau der Handrechnung entsprechen. |
| **`explain` gibt vier Erklärungen auf einmal zurück** | Ein `shapiq.Explainer` gibt normalerweise **ein** `InteractionValues` zurück. Unserer gibt P, TU, AU und EU zusammen zurück (`UncertaintyExplanation`), weil alle vier dieselben Koalitionen nutzen. Das ist die einzige bewusste Abweichung von shapiq; sie steht so in den Typangaben und ist für ty markiert. |
| **MC Dropout Zeile für Zeile mit festem Seed** | probly zieht je Zeile eigene Dropout-Masken, im Stapel abhängig von der Position. So wäre v(S) eine Zufallszahl. Zeile für Zeile sieht jede Koalition dieselben 50 Masken. |
| **64-Bit-Gleitkomma** | Sonst weicht TU = AU + EU um etwa 1e-7 ab (bei EU um 0,01 nicht vernachlässigbar). |
| **Köpfe auch auf ausgeblendeten Bildern trainieren** („maskiert“) | Leichte Form des Ersatzmodells (Surrogat) aus ViT Shapley: Wer erklärt wird, sollte ausgeblendete Bilder kennen, sonst misst die Erklärung Verhalten außerhalb der Trainingsdaten. Wir vergleichen beide Varianten. |
| **Lernender Erklärer klein und auf eingefrorenen Tokens** | Covert et al. trainieren einen ganzen ViT als Erklärer (GPU). Unsere Version ist ein kleiner Transformer auf den ViT-Tokens und lernt in Minuten auf der CPU. Dafür erklärt er vier Größen gleichzeitig und nutzt Superpixel. |
| **Vorab festgelegte Fragen F1 bis F11 mit Regeln** | Damit wir nicht nachträglich die passende Geschichte suchen. Negative Ergebnisse werden genauso berichtet. F8 bis F11 wurden am 25.11. mit Keith abgestimmt, bevor etwas gerechnet war. |
| **Ground Truth mit selbst gebauten Testbildern** (03-3) | Bei echten Fotos weiß niemand, welcher Bildteil das Modell „wirklich“ unsicher macht. Bauen wir die Ursache selbst ein (Schachbrett, zwei Rassen, Chimäre), wissen wir, wo sie liegt, und können prüfen, ob die Erklärung sie findet. |
| **Jedes Testbild hat eine Kontrolle** | Jede Konstruktion verändert das Bild auch anders (Flicken, Klebekante). Die Kontrolle hat dieselbe Konstruktion ohne die Ursache, z. B. einen Kopf derselben Rasse. Nur der Unterschied zählt. |
| **Zwei Stufen: erst das Modell, dann die Erklärung** | Reagiert das Modell nicht auf die Ursache, kann auch keine Erklärung sie zeigen. Ohne diese Trennung hätten wir F8 als Fehler des Explainers gedeutet. |
| **Grobe Spieler mit exakten Werten** (2 oder 3 Teile, aus denen das Bild gebaut ist) | Bei 4 oder 8 Koalitionen rechnet shapiq exakt. Die Ground Truth wird so nicht durch eine Näherung verwischt, und die Rechnung kostet Sekunden. Dazu prüfen 16 SLIC-Superpixel, ob auch die normale Pipeline die Ursache findet. |
| **Quellbilder aus der Validierung, nur sicher erkannte** | Trainingsbilder kennt das Modell schon. Und nur wenn jeder Teil für sich klar „Rasse A“ sagt, ist die eingebaute Ursache die einzige. |
| **Vier Indizes auf denselben groben Spielen** (F12) | Die Indizes unterscheiden sich bei drei Spielern nur darin, wie viel vom Dreier-Term (Kopf × Körper × Hintergrund) sie dem Paar geben (k-SII und FSII ½, STII ⅓, Möbius nichts). So sieht man, ob ein Ergebnis an der Wahl des Index hängt. Exakt und ohne neue Transformer-Läufe. |
| **Feine Paare nach groben Teilen zusammenfassen** (F13) | Jedes Superpixel gehört zu dem Teil mit den meisten seiner Pixel; die Summe aller Paare Kopf-Superpixel × Körper-Superpixel ist das feine Gegenstück zum groben Paar. So prüft man die normale Pipeline, ohne ihr die Teile vorzugeben. |
| **Neue Regeln nicht nachträglich ändern** | F12 und F13 zeigten, dass „größer als null“ eine schwache Regel ist. Wir berichten sie trotzdem so, wie sie vorher festgelegt war, und nehmen die Lehre („gegen die eigene Kontrolle messen“) für künftige Fragen mit. |

## Ergebnisse

| | Frage (vorab festgelegt) | Ergebnis | Kennzahl | Notebook |
|---|---|---|---|---|
| F1 | Gemischtes Ensemble hat ≥ 2× EU des ViT-Ensembles | **gestützt** | 0,037 gegenüber 0,015 nats (2,5×); 73 % zwischen den Netzen | 03-0 |
| F2 | Maskiert trainierte Köpfe: ≥ 5 Prozentpunkte mehr Accuracy auf ausgeblendeten Bildern | **nicht gestützt** | +4,1 Prozentpunkte; aber NLL 0,96 → 0,77 und EU auf ausgeblendeten Bildern deutlich kleiner | 03-0 |
| F3 | EU liegt überproportional auf dem Tier | **gestützt** | Median +10,5 Prozentpunkte (P +16,7, TU +15,0) | 03-1 |
| F4 | SLIC lenkt P stärker auf das Tier als ein 4 × 4-Raster | **gestützt**, aber kleiner Effekt | Median +1,4 Prozentpunkte | 03-1 |
| F5 | Übermalen statt Weglassen ändert die EU-Karte deutlich | **gestützt** | Median ρ EU = 0,35 (P: 0,72); v(∅) von P: 0,50 statt 0,03 | 03-1 |
| F6 | Maskiertes Training ändert die EU-Karte deutlich | **nicht gestützt** | Median ρ = 0,71 | 03-1 |
| F7 | Lernender Erklärer mindestens so genau wie KernelSHAP 256 (alle vier Größen) | **nicht gestützt** | relativer Fehler P 0,62 / TU 0,52 / AU 0,49 / EU 0,89 gegen 0,46 / 0,56 / 0,61 / 0,48 | 03-2 |
| F8 | Feine EU-Karte (16 SLIC) setzt ein fremdes Schachbrett in ≥ 50 % der Bilder auf Platz 1, öfter als bei der Kontrolle | **nicht gestützt** | Treffer 20 % (Kontrolle 15 %, Zufall 6 %); das Modell reagiert kaum (ΔEU +0,0003 nats) | 03-3 |
| F9 | Zwei Rassen in einem Bild: AU-Paar links × rechts in ≥ 80 % positiv und über der Kontrolle | **gestützt** | positiv 100 %; Median +3,584 gegen +3,036 (auch die Kontrolle ist immer positiv) | 03-3 |
| F10 | Chimäre: EU-Paar Kopf × Körper in ≥ 2/3 positiv und über der Kontrolle | **gestützt** | positiv 73 %; Median +0,1042 gegen −0,0019 | 03-3 |
| F11 | EU-Anteil am Paar bei Chimären größer als bei zwei Rassen | **nicht gestützt** | Median 0,039 gegen 0,065 | 03-3 |
| F12 | Chimäre: EU-Paar Kopf × Körper bei **allen vier** Indizes (k-SII, STII, FSII, Möbius; exakt) in ≥ 2/3 positiv und über der Kontrolle | **nicht gestützt** | positiv: k-SII und FSII 73 %, STII 60 %, Möbius 50 %; Median bei allen über der Kontrolle; größer als die eigene Kontrolle bei 73 bis 77 % | 03-3 |
| F13 | Chimäre mit 16 SLIC-Superpixeln: Summe der EU-Paare Kopf × Körper in ≥ 2/3 positiv und über der Kontrolle | **nicht gestützt** | positiv 50 % (Kontrolle 10 %); Median +0,0152 gegen −0,1717; größer als die eigene Kontrolle bei 85 % | 03-3 |

F12 und F13 wurden am 25.11. gegen 22:10 mit Keith abgestimmt, bevor sie gerechnet wurden; das Ergebnis von F10 war da
schon bekannt.

Weitere Zahlen:
- **Vorhersage:** ViT allein 93,2 %, gemischtes Ensemble 95,0 % Accuracy (Dan mit VGG16: 89,9 bis 90,1 %).
- **Fehlererkennung:** Error-AUROC von `1 − max(p)` 0,91 bis 0,93, von EU 0,87 bis 0,91. EU erkennt Fehler schlechter als `1 − max(p)`.
- **Methoden:** EU-Karten von `Ensemble ViT` und `MC Dropout ViT` (beide nur ViT) zeigen auf dieselben Superpixel
  (ρ = 0,87); das gemischte Ensemble zeigt teilweise woanders hin (ρ = 0,53 bis 0,62).
- **Vergleich mit Dan (F3, EU):** Dans VGG16-Reihe (3 × 3-Raster, Weichzeichner) kommt in unserer Zählweise
  (Tier + Rand) auf +11,1 Prozentpunkte, wir auf +10,5. Seine +31,3 gelten für ein anderes Spiel mit drei Spielern
  (Tier, Hintergrund, Rand) und sind nicht vergleichbar.
- **Lernender Erklärer:** eine Erklärung in unter 0,1 s (fast nur der eine Lauf des ganzen Bildes durch beide Netze)
  statt etwa 20 s für KernelSHAP 256 (256 solcher Läufe; die Zeiten hängen vom Rechner ab, das Verhältnis von etwa 250
  nicht); Gleichverteilung als Vergleich: 0,76 / 0,63 / 0,60 / 0,95.

## Was wir daraus lernen

**Zur Leitfrage**

1. **Wo:** Unsicherheit wird vor allem durch das Tier abgebaut (in den Beispielen oft durch das Gesicht). Die Karten
   für TU und AU sind fast überall blau: Einen Bildteil zu sehen, senkt die Unsicherheit. Das fehlende Wissen (EU) ist
   uneinheitlicher: Fast jeder zweite Bildteil **erhöht** es.
2. **Rauschen oder fehlendes Wissen:** Auf vollständigen Bildern ist der größte Teil der Unsicherheit aleatorisch
   (das passt dazu, dass sich manche Rassen sehr ähnlich sehen). Fehlendes Wissen ist klein, hängt aber stark an
   **Kombinationen** von Bildteilen und daran, **welche Netze** im Ensemble sind.
3. **Wie sicher ist die Antwort selbst?** Jede Unsicherheits-Erklärung hängt an zwei Entscheidungen, die nicht aus den
   Daten kommen: wie Bildteile entfernt werden (Weglassen oder Übermalen, F5) und welche Mitglieder das Ensemble hat
   (F1, Methodenvergleich). Erklärungen der Unsicherheit reagieren darauf stärker als Erklärungen der Vorhersage.
   Das haben wir auf zwei Datensätzen mit zwei Modellfamilien gesehen (im Vorversuch mit Hautbildern und CNNs, hier
   mit Oxford Pets und ViTs).
4. **Grenze von EU:** Die gefährlichsten Fehler sind sichere Verwechslungen. Dort sind sich alle Mitglieder einig, und
   EU ist klein. Solche Fehler findet hier keines unserer Maße. Für „Soll ein Mensch nachsehen?“ ist `1 − max(p)`
   im Mittel trotzdem besser als EU (Error-AUROC 0,91 bis 0,93 gegen 0,87 bis 0,91).
5. **Ground Truth (03-3):** Bei einer Kombination, die das Modell nie gesehen hat (Chimäre), findet die Erklärung das
   fehlende Wissen dort, wo wir es eingebaut haben, und zwar als **Paar** Kopf × Körper. Belastbar ist das im
   **Vergleich mit einer Kontrolle**: Mit jedem Index und auch mit 16 Superpixeln ist das Paar bei drei von vier
   Chimären größer als bei der Kontrolle. Ob ein einzelnes Paar über null liegt, hängt dagegen am Index (Möbius: nur bei
   der Hälfte) und an den Spielern (fein: bei der Hälfte). Ein fremdes Muster landet
   dagegen in der AU: Alle Mitglieder sagen gemeinsam „keine Ahnung“ (nur der Flicken sichtbar: TU 3,36, davon AU 3,18).
   Und die Zerlegung trennt „Widerspruch“ (zwei Rassen) nicht von „nie gesehen“ (Chimäre); beide erhöhen AU und EU.
   **Was als AU und was als EU zählt, entscheiden das Modell und seine Mitglieder, nicht unsere Geschichte über das Bild.**

**Für unser Projekt**

- **Explainer für jede probly-Methode:** erfüllt für Ensembles, MC Dropout und gemischte Ensembles. Gemischte Ensembles
  brauchen einen Umweg (`FeatureSlice`), weil `probly.method.ensemble` nur Kopien eines Modells baut. Das ist ein
  konkreter Punkt für die Betreuer und ein möglicher kleiner Beitrag zu probly.
- **Architektur:** Koalitionen auswerten (teuer) und Unsicherheit messen (billig) sind getrennte Bausteine
  (`CoalitionFeatures` und `UncertaintyGame`). Dadurch kosten weitere Methoden und Größen fast nichts, und Shapley-Werte
  gibt es zu k-SII gratis dazu.
- **Standard-Entfernung bei ViTs: Tokens weglassen.** Übermalen lässt Information durch und verändert die
  Unsicherheits-Karten.
- **Interaktionen lohnen sich für EU:** Bei EU sind die Paar-Beiträge zusammen dreimal so groß wie die Einzelbeiträge
  (bei P anderthalbmal). Toms Wahl von k-SII bis Ordnung 2 ist damit inhaltlich begründet.
- **Prüfstand für den Explainer (03-3):** Testbilder mit eingebauter Ursache, jeweils mit Kontrolle, exakte Spiele mit
  wenigen Spielern und zwei Stufen (erst das Modell, dann die Erklärung). Der Prüfstand funktioniert mit jeder
  probly-Methode; der nächste Schritt ist, ihn für weitere Methoden laufen zu lassen (Idee B1). Als Regel für neue
  Fragen eignet sich „größer als bei der eigenen Kontrolle“ besser als „größer als null“ (F12, F13).
- **Mehr shapiq:** Der Explainer rechnet jeden exakten Index (k-SII, STII, FSII, Möbius, …) und als Näherung k-SII, SII,
  FSII und Shapley-Werte, alle aus denselben Koalitionen. k-SII bleibt Standard: Es enthält die Shapley-Werte, ist bei
  drei Spielern genau gleich FSII und hängt weniger als Möbius an sehr stark ausgeblendeten Bildern. Für Abbildungen
  nutzen wir auch shapiqs eigene Plots (`upset_plot`, `force_plot`, `si_graph_plot`).
- **Code-Qualität:** `ruff` und `ty` laufen jetzt auch für diesen Ordner (eigene `ruff.toml` und `ty.toml`). Die 41
  alten Typfehler sind behoben; nur eine bewusste Abweichung von shapiq ist markiert (siehe Entscheidungen).
- **Mögliches Endthema:** „Amortisierte Unsicherheitserklärungen“ (ViT Shapley für P, TU, AU, EU mit Superpixeln).
  Der Prototyp zeigt, dass es für TU und AU schon funktioniert; für P und EU braucht er mehr Trainingsdaten.
- **Für die Präsentation** (in `figures/`): `ablauf.png` (der Ablauf), `fall_falsch.png` (sicherer Fehler, kaum EU),
  `weglassen_gegen_uebermalen.png`, `methoden_eu.png`, `vit_shapley_genauigkeit_zeit.png`, `vit_shapley_lernkurve.png`,
  `ground_truth_chimaere.png` (die EU-Karte färbt den Körper der Chimäre rot), `ground_truth_paare.png` (Test gegen
  Kontrolle), `ground_truth_testbilder.png`, `ground_truth_muster.png`, `ground_truth_indizes.png` (ein Ergebnis, vier
  Indizes), `ground_truth_chimaere_fein.png` (16 Superpixel), `ground_truth_upset.png` und `ground_truth_si_graph.png`
  (shapiq-Plots der Chimäre).

**Nächste Schritte**

- Auf einem schnelleren Rechner mehr Trainings-Koalitionen erzeugen (`settings.MASKED_PER_IMAGE_TRAIN = 32`) und alle
  drei Notebooks neu ausführen (die Einstellung ändert auch die maskierten Köpfe, also alle Ergebnisse). Die Lernkurve
  lässt bessere Werte für P und EU erwarten. Außerdem: gepaarte Koalitionen wie bei Covert et al. (S zusammen mit
  seinem Komplement ziehen).
- Den Ground-Truth-Prüfstand (03-3) mit weiteren probly-Methoden laufen lassen: Zeigen auch MC Dropout und eine Methode
  ohne Mitglieder (z. B. Evidential) das Chimären-Paar? Die groben Spiele kosten nur Sekunden.
- Mit den Betreuern klären: Ort im Code (D4), ob gemischte Ensembles als probly-Methode gewünscht sind, und welche
  Entfernung Standard sein soll.

## Grenzen

- Alle Netze bleiben eingefroren; nur kleine Köpfe werden trainiert. Ein Satz Seeds, MC Dropout nur mit p = 0,5.
- Erklärungen sind Näherungen (Budget 1000 von 65.536 Koalitionen); die Referenz in 03-2 ist KernelSHAP mit 2048.
- 37 Bilder für die Leitfrage (eins je Rasse), 13 für die teuren Vergleiche. Die Schwellen der Fragen sind einfache,
  vorher gesetzte Regeln, keine statistischen Tests.
- Der Tier-Anteil nutzt die Trimaps von Oxford Pets (Rand zählt zum Tier) und verteilt jeden Shapley-Wert gleichmäßig
  auf sein Superpixel.
- Der lernende Erklärer sieht nur 8 Koalitionen je Trainingsbild und arbeitet auf gemittelten Tokens je Superpixel;
  er ist ein Prototyp, kein fertiges Verfahren.
- Dans Zahlen stammen aus seinen Notebooks (VGG16, 3 Köpfe); unsere Ensembles haben 6 Mitglieder.
- Ground Truth (03-3): 30 Testbilder je Test (je 20 beim Muster und bei der Chimäre fein erklärt), nur die
  Hauptmethode. Die Testbilder sind künstlich (Klebekanten, Hintergrund im Kopf-Rahmen); die Kontrollen gleichen das
  aus, aber nicht ganz. Die Validierungsbilder dienten beim Training der Köpfe dem frühen Stoppen. F12 und F13 wurden
  festgelegt, als F10 schon bekannt war. Möbius-Terme und Paare hängen an stark ausgeblendeten Bildern, die das Modell
  kaum kennt (nur Hintergrund sichtbar: EU 0,32 nats).

## Protokoll

### 25./26.11.2026 · Mehr shapiq in 03-3, Typfehler behoben, Vorversuch zusammengefasst

- **Was (shapiq):** 03-3 hat drei neue Abschnitte. F12 rechnet dieselben groben Spiele exakt mit k-SII, STII, FSII und
  Möbius; 7.1 zeigt die EU je Koalition und die Möbius-Terme. F13 erklärt 20 Chimären und 20 Kontrollen mit 16
  SLIC-Superpixeln (k-SII, dazu FSII aus denselben Koalitionen). Abschnitt 9 zeigt die Chimäre mit shapiqs `upset_plot`,
  `force_plot` und `si_graph_plot`. Neu im Paket: `UncertaintyExplainer` kann FSII schätzen (`RegressionFSII`) und
  jeden Index exakt rechnen; `gt.superpixel_parts`, `gt.group_pair_sum`, `plot.superpixel_patches`. Vier neue
  Abbildungen.
- **Was (Typfehler):** Alle 41 Fehler behoben (`ty` 0.0.82 aus dem Repository und 0.0.110: 0 Fehler in
  `vit_uncertainty/`, `tests/` und `scripts/`). Neue `ty.toml` für den Ordner. Die drei Design-Punkte: (1) das
  gemischte Ensemble ist jetzt eine selbst gebaute `nn.ModuleList`, statt Kopien von probly auszutauschen; (2) Köpfe
  werden über die Registry von probly als „logit classifier“ angemeldet, statt mit dem Schlüsselwort
  `predictor_type`, das ty nur mit generierten Stubs kennt; (3) `explain` und `explain_function` geben bewusst eine
  `UncertaintyExplanation` zurück (die einzige markierte Ausnahme). Dazu: `TypedDict` für Einstellungen
  (`BackboneSpec`, `HeadConfig`), eine Unterklasse `DiskCoalitionFeatures` statt nachträglich angehängter Attribute,
  Prüfungen auf `None`, und die Zerlegung von probly wird auf `AdditiveDecomposition` geprüft (der Explainer braucht
  TU = AU + EU).
- **Neue Tests (40 statt 31):** zwei für den neuen Weg zu probly (gegen die Handrechnung und gegen
  `predictor_type="logit_classifier"`), vier für die Indizes (z. B. die Anteile ½ und ⅓ am Dreier-Term), drei für die
  neuen Hilfsfunktionen.
- **Was (Text):** Die DermaMNIST-Stellen in README, 03-0, 03-1 und zwei Docstrings sind durch den Absatz „Vorversuch“
  ersetzt.
- **Warum:** mehr shapiq für die Bewertung (Keiths Wunsch), Code-Qualität (Aufgabe S2), und die Grenze aus 03-3 („k-SII
  gibt dem Paar einen Teil des Dreier-Terms“) prüfen.
- **Ablauf:** F12 und F13 mit Keith abgestimmt (25.11., gegen 22:10), **dann** gerechnet. Die Regeln wurden nach der
  Rechnung nicht geändert.
- **Geprüft:** Python 3.13, torch 2.15.0 (PyPI-Build, läuft auf der CPU), timm 1.0.31, shapiq 1.8.0, scikit-learn
  1.9.1, numpy 2.5.3, probly `6508e31`. Alle Caches wurden neu gebaut, alle vier Notebooks von oben ausgeführt. Alle
  Zahlen von 03-0 bis 03-3, die es schon gab, sind **gleich** geblieben; die alten Abbildungen von 03-3 sind sogar
  Byte für Byte gleich. Anders sind nur Zeiten (in 03-2 auch die daraus berechneten Sekunden je Erklärung; das
  Verhältnis von etwa 250 bleibt), die Versionszeilen und in einer Rohausgabe von 03-1 Stellen ab der 9.
  Nachkommastelle (Rechenrauschen). Damit ist auch gezeigt, dass die Typ-Korrekturen nichts am Verhalten ändern.
  `pytest tests` (40 Tests grün), `ruff check`, `ruff format --check`, `ty check`.
- **Ergebnis:** F12 und F13 nicht gestützt. Der Abstand zur Kontrolle ist in allen Varianten stabil (73 bis 85 %), das
  Vorzeichen des Paares nicht. Korrigiert: Im Beispiel aus 3.1 ist 0,1731 die EU des ganzen Bildes, nicht von Kopf und
  Körper allein (0,0190).
- **Offen:** Prüfstand mit weiteren probly-Methoden (B1).

### 25.11.2026 · Ground Truth (Notebook 03-3)

- **Was:** Neues Modul `vit_uncertainty/ground_truth.py` (Testbilder mit eingebauter Ursache, grobe Spieler,
  Kennzahlen), `OxfordPets.load_head_box` in `data.py` (Kopf-Rahmen aus `annotations/xmls`, auf den 224-Pixel-Ausschnitt
  gebracht), 11 neue Tests, Notebook `keith_03-3_ground-truth.ipynb`, vier Abbildungen `figures/ground_truth_*.png`.
  Sonst wurde keine Datei geändert.
- **Warum:** Größte Lücke für die Verteidigung war, dass wir nicht zeigen konnten, ob die Karten stimmen (Übergabe, Idee
  A1).
- **Ablauf:** Plan und Regeln F8 bis F11 mit Keith abgestimmt (25.11., 16:42), **dann** gerechnet. Zwei Regeln für die
  Bildauswahl kamen vor der Rechnung dazu: Körper ≥ 10 % des Bildes (Nahaufnahmen haben keinen Körper) und jeder grobe
  Spieler braucht Tokens in beiden Netzen.
- **Geprüft:** Python 3.13, torch 2.15 (CPU), timm 1.0.31, shapiq 1.8.0, scikit-learn 1.9.1, probly `6508e31`. Alle
  Caches wurden neu gebaut; Notebook 03-0 liefert dabei **dieselben Zahlen** wie am 24.11. (nur die gemessenen
  Millisekunden je Bild schwanken). `pytest tests` (31 Tests grün), `ruff check` und `ruff format --check`; `ty` findet
  im neuen Code keinen Fehler (die 41 alten Typfehler sind unverändert). Notebook 03-3 zweimal von oben ausgeführt
  (zweiter Lauf aus dem Cache, gleiche Zahlen). Alle Zahlen in dieser README und in der Deutung von 03-3 mit den Ausgaben
  abgeglichen.
- **Ergebnis:** F9 und F10 gestützt, F8 und F11 nicht gestützt (siehe Ergebnisse).
- **Offen:** Prüfstand mit weiteren probly-Methoden (B1).

### 24.11.2026 · Reihe 03 angelegt

- **Was:** Neuer Ordner `keith_03_vit_uncertainty/` mit dem Paket `vit_uncertainty` (11 Module), 20 Tests, drei
  Skripten, drei ausgeführten Notebooks, Abbildungen und dieser README. Keine Datei außerhalb des Ordners wurde geändert.
- **Warum:** Prototyp einer shapiq-Erweiterung für Bilder: Modell → probly → shapiq-Explainer → Erklärung von Vorhersage
  und Unsicherheit, plus ViT Shapley für Unsicherheit. Auf Dans Datensatz, mit den Erkenntnissen aus dem Vorversuch und
  Toms Superpixel-Vergleich.
- **Geprüft:** Python 3.13, torch 2.15 (CPU), timm 1.0, shapiq 1.8.0, probly `6508e31` (Stand des Fork-`main`).
  `pytest tests` (20 Tests grün), `ruff check` und `ruff format --check` mit den Regeln des Repos (lokale `ruff.toml`).
  Alle drei Notebooks frisch von oben ausgeführt. Prüfungen in den Notebooks: probly gegen Handrechnung, TU = AU + EU,
  Effizienz aller Erklärungen, φ(TU) = φ(AU) + φ(EU) für Shapley-Werte und k-SII-Paare. Alle Zahlen in dieser README
  und in den Deutungen der Notebooks mit den Ausgaben abgeglichen, Dans Vergleichswerte mit seinen Notebooks.
- **Unterwegs gefunden und gelöst:** SLIC liefert auf 224-Pixel-Bildern zu wenige Superpixel (jetzt genau 16);
  MC-Dropout-Werte hingen vom Stapel ab (jetzt Zeile für Zeile mit festen Masken); der erste lernende Erklärer war zu
  langsam und lernte auswendig (jetzt Spieler-Tokens und ein kleines Netz).
- **Offen:** siehe „Nächste Schritte“.
