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
  ein Ensemble aus zwei verschiedenen Transformern. Effizienz und φ(TU) = φ(AU) + φ(EU) gelten, 31 Tests sind grün.
- **Zwei verschiedene Netze zeigen mehr fehlendes Wissen.** Das gemischte Ensemble (ViT + DINOv2) hat 2,5-mal so viel
  EU wie Köpfe nur auf dem ViT; 73 % davon entstehen **zwischen** den Netzen. Das bestätigt, was wir bei DermaMNIST
  gesehen haben.
- **Tokens weglassen ist ehrlicher als Übermalen.** Ist alles ausgeblendet, sagt das Modell „keine Ahnung“
  (TU 3,45 nats, nahe am Maximum ln 37 ≈ 3,61). Ist alles weichgezeichnet, gibt das Modell der vorhergesagten Rasse
  im Mittel noch 50 %: Der Weichzeichner entfernt die Information gar nicht. Die EU-Karten unterscheiden sich stark
  (ρ = 0,35).
- **Fehlendes Wissen liegt eher auf dem Tier** (+10,5 Prozentpunkte über dem Flächenanteil; Dan mit VGG16 in derselben
  Zählweise: +11,1) und hängt stark an **Kombinationen** von Bildteilen: Bei EU sind die Paar-Beiträge zusammen dreimal
  so groß wie die Einzelbeiträge, bei P nur anderthalbmal.
- **EU erkennt sichere Fehler nicht.** Ein Staffordshire Bull Terrier wird mit 96 % als American Bulldog vorhergesagt,
  bei einer EU von nur 0,010 nats.
- **Ground Truth: Die Paare stimmen.** Auf Testbildern mit eingebauter Ursache (03-3) zeigt eine Chimäre (Kopf von
  Rasse A auf dem Körper von Rasse B) fehlendes Wissen genau als Paar Kopf × Körper (+0,104 gegen −0,002 bei der
  Kontrolle). Ein fremdes Schachbrett macht das Modell dagegen unsicher, aber als **Rauschen**: Alle Mitglieder sagen
  gemeinsam „keine Ahnung“. EU ist hier also kein Alarm für Fremdes.
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
| `vit_uncertainty/plot.py` | Vorhersage und Unsicherheit nebeneinander, Paare, Vergleiche |
| `vit_uncertainty/amortized.py` | `AmortizedUncertaintyExplainer`: ViT Shapley für Unsicherheit |
| `vit_uncertainty/features.py` | Caches für Merkmale, maskierte Trainingsdaten und Koalitionen |
| `vit_uncertainty/ground_truth.py` | Testbilder mit bekannter Ursache (Muster, zwei Rassen, Chimäre), grobe Spieler, Kennzahlen |
| `vit_uncertainty/settings.py` | alle festen Einstellungen an einem Ort |
| `tests/` | Tests mit kleinen Zufalls-Transformern (laufen in Sekunden, ohne Download) |
| `scripts/benchmark.py` | misst die Geschwindigkeit deines Rechners und schätzt alle Laufzeiten |
| `scripts/precompute.py` | rechnet alle Transformer-Läufe vorab (optional, z. B. über Nacht) |
| `scripts/make_pipeline_figure.py` | zeichnet die Ablauf-Abbildung `figures/ablauf.png` |
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
| Ground Truth (03-3): 40 feine Erklärungen (16 SLIC-Superpixel, Budget 1000) | ca. 75 min | einmalig, danach Cache in `artifacts/keith-vit-uncertainty/ground_truth/` |
| Lernender Erklärer: eine Erklärung | ca. 0,1 s: ein Lauf durch beide Netze, der Erklärer selbst höchstens etwa 1 ms | statt 1000 Läufen |

Laufen mehrere Rechnungen gleichzeitig, teilen sie sich die Kerne und jede wird entsprechend langsamer.

## Entscheidungen und warum

| Entscheidung | Warum |
|---|---|
| **Oxford-IIIT Pet mit Dans Aufteilung** | Gleiche Bilder, gleicher 224-Pixel-Ausschnitt, gleiche Kohorten wie in Dans VGG16-Reihe. So sind die Zahlen direkt vergleichbar. Dans Code wird nicht importiert; `data.py` baut die Aufteilung nach. |
| **Vision Transformer statt CNN** | Ein ViT verarbeitet das Bild als Tokens. Einen Bildteil kann man **weglassen** (Tokens ausblenden), statt ihn mit einer Farbe zu übermalen. Bei DermaMNIST hatte gerade das Übermalen die Erklärungen der Unsicherheit stark verändert. |
| **Zwei verschiedene Transformer** (ViT-S/16 überwacht, DINOv2-S/14 selbstüberwacht) | Aus DermaMNIST: Köpfe auf einem Netz sind sich fast immer einig, ihre EU ist fast null. Mitglieder, die das Bild verschieden „sehen“, zeigen fehlendes Wissen. |
| **Kleine „S“-Modelle, eingefroren** | Für eine Erklärung läuft jedes Bild hunderte Male durch beide Netze. Die S-Modelle sind etwa dreimal so schnell wie ViT-B und laufen auf einer CPU. Nur kleine Köpfe werden trainiert (Dans Einstellungen). |
| **Tokens per Attention-Maske ausblenden** | Für das CLS-Token ist das exakt dasselbe wie die Tokens zu löschen (getestet), lässt sich aber für viele Koalitionen in einem Stapel rechnen. |
| **16 SLIC-Superpixel, exakt 16** | Wie Toms Vergleich (16 Segmente, Kompaktheit 10). SLIC liefert auf 224-Pixel-Bildern meist nur 10 bis 14; wir erhöhen die Anfrage und verschmelzen dann die kleinsten, bis es genau 16 sind. So hat jedes Bild gleich viele Spieler und ein Budget bedeutet überall dasselbe. Jeder Token gehört zu dem Superpixel, das die meisten seiner Pixel bedeckt. |
| **k-SII bis Ordnung 2, Budget 1000** | Wie bei Tom. Einzelbeiträge und Paare; 1000 liegt deutlich über den 137 Unbekannten (1 + 16 + 120). |
| **Vier Spiele (P, TU, AU, EU) teilen sich alles** | Gleicher Seed → shapiq zieht dieselben Koalitionen; ein Cache sorgt dafür, dass jede Koalition nur einmal durch die Netze und nur einmal durch probly geht. |
| **Gemischtes Ensemble über `FeatureSlice`** | `probly.method.ensemble` kopiert ein Grundmodell. Unsere Mitglieder sitzen auf verschiedenen Netzen; jedes liest denselben Vektor und schneidet seinen Teil heraus. probly baut den Container, wir setzen die Köpfe ein. |
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

Weitere Zahlen:
- **Vorhersage:** ViT allein 93,2 %, gemischtes Ensemble 95,0 % Accuracy (Dan mit VGG16: 89,9 bis 90,1 %).
- **Fehlererkennung:** Error-AUROC von `1 − max(p)` 0,91 bis 0,93, von EU 0,87 bis 0,91. EU erkennt Fehler schlechter als `1 − max(p)`.
- **Methoden:** EU-Karten von `Ensemble ViT` und `MC Dropout ViT` (beide nur ViT) zeigen auf dieselben Superpixel
  (ρ = 0,87); das gemischte Ensemble zeigt teilweise woanders hin (ρ = 0,53 bis 0,62).
- **Vergleich mit Dan (F3, EU):** Dans VGG16-Reihe (3 × 3-Raster, Weichzeichner) kommt in unserer Zählweise
  (Tier + Rand) auf +11,1 Prozentpunkte, wir auf +10,5. Seine +31,3 gelten für ein anderes Spiel mit drei Spielern
  (Tier, Hintergrund, Rand) und sind nicht vergleichbar.
- **Lernender Erklärer:** eine Erklärung in etwa 0,1 s (fast nur der eine Lauf des ganzen Bildes durch beide Netze)
  statt etwa 28 s für KernelSHAP 256 (256 solcher Läufe); Gleichverteilung als Vergleich: 0,76 / 0,63 / 0,60 / 0,95.

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
   Das haben wir auf zwei Datensätzen mit zwei Modellfamilien gesehen (DermaMNIST mit CNNs, Oxford Pets mit ViTs).
4. **Grenze von EU:** Die gefährlichsten Fehler sind sichere Verwechslungen. Dort sind sich alle Mitglieder einig, und
   EU ist klein. Solche Fehler findet hier keines unserer Maße. Für „Soll ein Mensch nachsehen?“ ist `1 − max(p)`
   im Mittel trotzdem besser als EU (Error-AUROC 0,91 bis 0,93 gegen 0,87 bis 0,91).
5. **Ground Truth (03-3):** Bei einer Kombination, die das Modell nie gesehen hat (Chimäre), findet die Erklärung das
   fehlende Wissen genau dort, wo wir es eingebaut haben, und zwar als **Paar** Kopf × Körper. Ein fremdes Muster landet
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
- **Prüfstand für den Explainer (03-3):** Testbilder mit eingebauter Ursache, exakte Spiele mit wenigen Spielern und
  zwei Stufen (erst das Modell, dann die Erklärung). Der Prüfstand funktioniert mit jeder probly-Methode; der nächste
  Schritt ist, ihn für weitere Methoden laufen zu lassen (Idee B1).
- **Mögliches Endthema:** „Amortisierte Unsicherheitserklärungen“ (ViT Shapley für P, TU, AU, EU mit Superpixeln).
  Der Prototyp zeigt, dass es für TU und AU schon funktioniert; für P und EU braucht er mehr Trainingsdaten.
- **Für die Präsentation** (in `figures/`): `ablauf.png` (der Ablauf), `fall_falsch.png` (sicherer Fehler, kaum EU),
  `weglassen_gegen_uebermalen.png`, `methoden_eu.png`, `vit_shapley_genauigkeit_zeit.png`, `vit_shapley_lernkurve.png`,
  `ground_truth_chimaere.png` (die EU-Karte färbt den Körper der Chimäre rot), `ground_truth_paare.png` (Test gegen
  Kontrolle), `ground_truth_testbilder.png`, `ground_truth_muster.png`.

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
- Ground Truth (03-3): 30 Testbilder je Test (20 fein erklärt), nur die Hauptmethode. Die Testbilder sind künstlich
  (Klebekanten, Hintergrund im Kopf-Rahmen); die Kontrollen gleichen das aus, aber nicht ganz. Die Validierungsbilder
  dienten beim Training der Köpfe dem frühen Stoppen.

## Protokoll

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
  und Unsicherheit, plus ViT Shapley für Unsicherheit. Auf Dans Datensatz, mit den Erkenntnissen aus DermaMNIST und
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
