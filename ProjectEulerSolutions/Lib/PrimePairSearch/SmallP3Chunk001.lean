import ProjectEulerSolutions.Lib.PrimePairSearch.SmallP3Chunk000
import ProjectEulerSolutions.Lib.CliqueSearch

/-! Generated scan steps for the first small-domain branch. -/
namespace ProjectEulerSolutions.PrimePairSearch
open PrimePairCertificate CliqueSearch

def smallTail92 : List Nat := 5801 :: smallTail93
def smallCandidates92 : List Nat := [6263]
set_option maxRecDepth 2048 in
theorem smallCandidates92_checked : candidates relation 4 3 5801 smallTail93 smallBest.bound = smallCandidates92 := by decide

theorem smallScan92 : scan relation (search relation 3) 4 [3] 3 smallTail92 smallBest = smallBest := by
  rw [smallTail92, scan, if_neg (by decide), smallCandidates92_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail93
    (if 3 ≤ smallCandidates92.length then search relation 3 [3, 5801] smallCandidates92 5804 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan93

def smallTail91 : List Nat := 5573 :: smallTail92
def smallCandidates91 : List Nat := [8219]
set_option maxRecDepth 2048 in
theorem smallCandidates91_checked : candidates relation 4 3 5573 smallTail92 smallBest.bound = smallCandidates91 := by decide

theorem smallScan91 : scan relation (search relation 3) 4 [3] 3 smallTail91 smallBest = smallBest := by
  rw [smallTail91, scan, if_neg (by decide), smallCandidates91_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail92
    (if 3 ≤ smallCandidates91.length then search relation 3 [3, 5573] smallCandidates91 5576 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan92

def smallTail90 : List Nat := 5531 :: smallTail91
def smallCandidates90 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates90_checked : candidates relation 4 3 5531 smallTail91 smallBest.bound = smallCandidates90 := by decide

theorem smallScan90 : scan relation (search relation 3) 4 [3] 3 smallTail90 smallBest = smallBest := by
  rw [smallTail90, scan, if_neg (by decide), smallCandidates90_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail91
    (if 3 ≤ smallCandidates90.length then search relation 3 [3, 5531] smallCandidates90 5534 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan91

def smallTail89 : List Nat := 5521 :: smallTail90
def smallCandidates89 : List Nat := [7159]
set_option maxRecDepth 2048 in
theorem smallCandidates89_checked : candidates relation 4 3 5521 smallTail90 smallBest.bound = smallCandidates89 := by decide

theorem smallScan89 : scan relation (search relation 3) 4 [3] 3 smallTail89 smallBest = smallBest := by
  rw [smallTail89, scan, if_neg (by decide), smallCandidates89_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail90
    (if 3 ≤ smallCandidates89.length then search relation 3 [3, 5521] smallCandidates89 5524 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan90

def smallTail88 : List Nat := 5507 :: smallTail89
def smallCandidates88 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates88_checked : candidates relation 4 3 5507 smallTail89 smallBest.bound = smallCandidates88 := by decide

theorem smallScan88 : scan relation (search relation 3) 4 [3] 3 smallTail88 smallBest = smallBest := by
  rw [smallTail88, scan, if_neg (by decide), smallCandidates88_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail89
    (if 3 ≤ smallCandidates88.length then search relation 3 [3, 5507] smallCandidates88 5510 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan89

def smallTail87 : List Nat := 5449 :: smallTail88
def smallCandidates87 : List Nat := [6073]
set_option maxRecDepth 2048 in
theorem smallCandidates87_checked : candidates relation 4 3 5449 smallTail88 smallBest.bound = smallCandidates87 := by decide

theorem smallScan87 : scan relation (search relation 3) 4 [3] 3 smallTail87 smallBest = smallBest := by
  rw [smallTail87, scan, if_neg (by decide), smallCandidates87_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail88
    (if 3 ≤ smallCandidates87.length then search relation 3 [3, 5449] smallCandidates87 5452 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan88

def smallTail86 : List Nat := 5419 :: smallTail87
def smallCandidates86 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates86_checked : candidates relation 4 3 5419 smallTail87 smallBest.bound = smallCandidates86 := by decide

theorem smallScan86 : scan relation (search relation 3) 4 [3] 3 smallTail86 smallBest = smallBest := by
  rw [smallTail86, scan, if_neg (by decide), smallCandidates86_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail87
    (if 3 ≤ smallCandidates86.length then search relation 3 [3, 5419] smallCandidates86 5422 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan87

def smallTail85 : List Nat := 5381 :: smallTail86
def smallCandidates85 : List Nat := [5507, 6599]
set_option maxRecDepth 2048 in
theorem smallCandidates85_checked : candidates relation 4 3 5381 smallTail86 smallBest.bound = smallCandidates85 := by decide

theorem smallScan85 : scan relation (search relation 3) 4 [3] 3 smallTail85 smallBest = smallBest := by
  rw [smallTail85, scan, if_neg (by decide), smallCandidates85_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail86
    (if 3 ≤ smallCandidates85.length then search relation 3 [3, 5381] smallCandidates85 5384 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan86

def smallTail84 : List Nat := 5323 :: smallTail85
def smallCandidates84 : List Nat := [5521, 6037]
set_option maxRecDepth 2048 in
theorem smallCandidates84_checked : candidates relation 4 3 5323 smallTail85 smallBest.bound = smallCandidates84 := by decide

theorem smallScan84 : scan relation (search relation 3) 4 [3] 3 smallTail84 smallBest = smallBest := by
  rw [smallTail84, scan, if_neg (by decide), smallCandidates84_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail85
    (if 3 ≤ smallCandidates84.length then search relation 3 [3, 5323] smallCandidates84 5326 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan85

def smallTail83 : List Nat := 5281 :: smallTail84
def smallCandidates83 : List Nat := [5869, 7159]
set_option maxRecDepth 2048 in
theorem smallCandidates83_checked : candidates relation 4 3 5281 smallTail84 smallBest.bound = smallCandidates83 := by decide

theorem smallScan83 : scan relation (search relation 3) 4 [3] 3 smallTail83 smallBest = smallBest := by
  rw [smallTail83, scan, if_neg (by decide), smallCandidates83_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail84
    (if 3 ≤ smallCandidates83.length then search relation 3 [3, 5281] smallCandidates83 5284 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan84

def smallTail82 : List Nat := 5171 :: smallTail83
def smallCandidates82 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates82_checked : candidates relation 4 3 5171 smallTail83 smallBest.bound = smallCandidates82 := by decide

theorem smallScan82 : scan relation (search relation 3) 4 [3] 3 smallTail82 smallBest = smallBest := by
  rw [smallTail82, scan, if_neg (by decide), smallCandidates82_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail83
    (if 3 ≤ smallCandidates82.length then search relation 3 [3, 5171] smallCandidates82 5174 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan83

def smallTail81 : List Nat := 5099 :: smallTail82
def smallCandidates81 : List Nat := [5507, 7649]
set_option maxRecDepth 2048 in
theorem smallCandidates81_checked : candidates relation 4 3 5099 smallTail82 smallBest.bound = smallCandidates81 := by decide

theorem smallScan81 : scan relation (search relation 3) 4 [3] 3 smallTail81 smallBest = smallBest := by
  rw [smallTail81, scan, if_neg (by decide), smallCandidates81_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail82
    (if 3 ≤ smallCandidates81.length then search relation 3 [3, 5099] smallCandidates81 5102 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan82

def smallTail80 : List Nat := 5059 :: smallTail81
def smallCandidates80 : List Nat := [5323]
set_option maxRecDepth 2048 in
theorem smallCandidates80_checked : candidates relation 4 3 5059 smallTail81 smallBest.bound = smallCandidates80 := by decide

theorem smallScan80 : scan relation (search relation 3) 4 [3] 3 smallTail80 smallBest = smallBest := by
  rw [smallTail80, scan, if_neg (by decide), smallCandidates80_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail81
    (if 3 ≤ smallCandidates80.length then search relation 3 [3, 5059] smallCandidates80 5062 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan81

def smallTail79 : List Nat := 5051 :: smallTail80
def smallCandidates79 : List Nat := [7253]
set_option maxRecDepth 2048 in
theorem smallCandidates79_checked : candidates relation 4 3 5051 smallTail80 smallBest.bound = smallCandidates79 := by decide

theorem smallScan79 : scan relation (search relation 3) 4 [3] 3 smallTail79 smallBest = smallBest := by
  rw [smallTail79, scan, if_neg (by decide), smallCandidates79_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail80
    (if 3 ≤ smallCandidates79.length then search relation 3 [3, 5051] smallCandidates79 5054 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan80

def smallTail78 : List Nat := 4919 :: smallTail79
def smallCandidates78 : List Nat := [6599]
set_option maxRecDepth 2048 in
theorem smallCandidates78_checked : candidates relation 4 3 4919 smallTail79 smallBest.bound = smallCandidates78 := by decide

theorem smallScan78 : scan relation (search relation 3) 4 [3] 3 smallTail78 smallBest = smallBest := by
  rw [smallTail78, scan, if_neg (by decide), smallCandidates78_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail79
    (if 3 ≤ smallCandidates78.length then search relation 3 [3, 4919] smallCandidates78 4922 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan79

def smallTail77 : List Nat := 4729 :: smallTail78
def smallCandidates77 : List Nat := [7963]
set_option maxRecDepth 2048 in
theorem smallCandidates77_checked : candidates relation 4 3 4729 smallTail78 smallBest.bound = smallCandidates77 := by decide

theorem smallScan77 : scan relation (search relation 3) 4 [3] 3 smallTail77 smallBest = smallBest := by
  rw [smallTail77, scan, if_neg (by decide), smallCandidates77_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail78
    (if 3 ≤ smallCandidates77.length then search relation 3 [3, 4729] smallCandidates77 4732 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan78

def smallTail76 : List Nat := 4583 :: smallTail77
def smallCandidates76 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates76_checked : candidates relation 4 3 4583 smallTail77 smallBest.bound = smallCandidates76 := by decide

theorem smallScan76 : scan relation (search relation 3) 4 [3] 3 smallTail76 smallBest = smallBest := by
  rw [smallTail76, scan, if_neg (by decide), smallCandidates76_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail77
    (if 3 ≤ smallCandidates76.length then search relation 3 [3, 4583] smallCandidates76 4586 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan77

def smallTail75 : List Nat := 4253 :: smallTail76
def smallCandidates75 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates75_checked : candidates relation 4 3 4253 smallTail76 smallBest.bound = smallCandidates75 := by decide

theorem smallScan75 : scan relation (search relation 3) 4 [3] 3 smallTail75 smallBest = smallBest := by
  rw [smallTail75, scan, if_neg (by decide), smallCandidates75_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail76
    (if 3 ≤ smallCandidates75.length then search relation 3 [3, 4253] smallCandidates75 4256 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan76

def smallTail74 : List Nat := 4159 :: smallTail75
def smallCandidates74 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates74_checked : candidates relation 4 3 4159 smallTail75 smallBest.bound = smallCandidates74 := by decide

theorem smallScan74 : scan relation (search relation 3) 4 [3] 3 smallTail74 smallBest = smallBest := by
  rw [smallTail74, scan, if_neg (by decide), smallCandidates74_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail75
    (if 3 ≤ smallCandidates74.length then search relation 3 [3, 4159] smallCandidates74 4162 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan75

def smallTail73 : List Nat := 4019 :: smallTail74
def smallCandidates73 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates73_checked : candidates relation 4 3 4019 smallTail74 smallBest.bound = smallCandidates73 := by decide

theorem smallScan73 : scan relation (search relation 3) 4 [3] 3 smallTail73 smallBest = smallBest := by
  rw [smallTail73, scan, if_neg (by decide), smallCandidates73_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail74
    (if 3 ≤ smallCandidates73.length then search relation 3 [3, 4019] smallCandidates73 4022 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan74

def smallTail72 : List Nat := 3931 :: smallTail73
def smallCandidates72 : List Nat := [5449]
set_option maxRecDepth 2048 in
theorem smallCandidates72_checked : candidates relation 4 3 3931 smallTail73 smallBest.bound = smallCandidates72 := by decide

theorem smallScan72 : scan relation (search relation 3) 4 [3] 3 smallTail72 smallBest = smallBest := by
  rw [smallTail72, scan, if_neg (by decide), smallCandidates72_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail73
    (if 3 ≤ smallCandidates72.length then search relation 3 [3, 3931] smallCandidates72 3934 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan73

def smallTail71 : List Nat := 3923 :: smallTail72
def smallCandidates71 : List Nat := [4919, 8237]
set_option maxRecDepth 2048 in
theorem smallCandidates71_checked : candidates relation 4 3 3923 smallTail72 smallBest.bound = smallCandidates71 := by decide

theorem smallScan71 : scan relation (search relation 3) 4 [3] 3 smallTail71 smallBest = smallBest := by
  rw [smallTail71, scan, if_neg (by decide), smallCandidates71_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail72
    (if 3 ≤ smallCandidates71.length then search relation 3 [3, 3923] smallCandidates71 3926 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan72

def smallTail70 : List Nat := 3911 :: smallTail71
def smallCandidates70 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates70_checked : candidates relation 4 3 3911 smallTail71 smallBest.bound = smallCandidates70 := by decide

theorem smallScan70 : scan relation (search relation 3) 4 [3] 3 smallTail70 smallBest = smallBest := by
  rw [smallTail70, scan, if_neg (by decide), smallCandidates70_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail71
    (if 3 ≤ smallCandidates70.length then search relation 3 [3, 3911] smallCandidates70 3914 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan71

def smallTail69 : List Nat := 3769 :: smallTail70
def smallCandidates69 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates69_checked : candidates relation 4 3 3769 smallTail70 smallBest.bound = smallCandidates69 := by decide

theorem smallScan69 : scan relation (search relation 3) 4 [3] 3 smallTail69 smallBest = smallBest := by
  rw [smallTail69, scan, if_neg (by decide), smallCandidates69_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail70
    (if 3 ≤ smallCandidates69.length then search relation 3 [3, 3769] smallCandidates69 3772 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan70

def smallTail68 : List Nat := 3637 :: smallTail69
def smallCandidates68 : List Nat := [5839, 6871]
set_option maxRecDepth 2048 in
theorem smallCandidates68_checked : candidates relation 4 3 3637 smallTail69 smallBest.bound = smallCandidates68 := by decide

theorem smallScan68 : scan relation (search relation 3) 4 [3] 3 smallTail68 smallBest = smallBest := by
  rw [smallTail68, scan, if_neg (by decide), smallCandidates68_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail69
    (if 3 ≤ smallCandidates68.length then search relation 3 [3, 3637] smallCandidates68 3640 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan69

def smallTail67 : List Nat := 3469 :: smallTail68
def smallCandidates67 : List Nat := [5839]
set_option maxRecDepth 2048 in
theorem smallCandidates67_checked : candidates relation 4 3 3469 smallTail68 smallBest.bound = smallCandidates67 := by decide

theorem smallScan67 : scan relation (search relation 3) 4 [3] 3 smallTail67 smallBest = smallBest := by
  rw [smallTail67, scan, if_neg (by decide), smallCandidates67_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail68
    (if 3 ≤ smallCandidates67.length then search relation 3 [3, 3469] smallCandidates67 3472 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan68

def smallTail66 : List Nat := 3461 :: smallTail67
def smallCandidates66 : List Nat := [4583]
set_option maxRecDepth 2048 in
theorem smallCandidates66_checked : candidates relation 4 3 3461 smallTail67 smallBest.bound = smallCandidates66 := by decide

theorem smallScan66 : scan relation (search relation 3) 4 [3] 3 smallTail66 smallBest = smallBest := by
  rw [smallTail66, scan, if_neg (by decide), smallCandidates66_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail67
    (if 3 ≤ smallCandidates66.length then search relation 3 [3, 3461] smallCandidates66 3464 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan67

def smallTail65 : List Nat := 3301 :: smallTail66
def smallCandidates65 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates65_checked : candidates relation 4 3 3301 smallTail66 smallBest.bound = smallCandidates65 := by decide

theorem smallScan65 : scan relation (search relation 3) 4 [3] 3 smallTail65 smallBest = smallBest := by
  rw [smallTail65, scan, if_neg (by decide), smallCandidates65_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail66
    (if 3 ≤ smallCandidates65.length then search relation 3 [3, 3301] smallCandidates65 3304 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan66

def smallTail64 : List Nat := 3119 :: smallTail65
def smallCandidates64 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates64_checked : candidates relation 4 3 3119 smallTail65 smallBest.bound = smallCandidates64 := by decide

theorem smallScan64 : scan relation (search relation 3) 4 [3] 3 smallTail64 smallBest = smallBest := by
  rw [smallTail64, scan, if_neg (by decide), smallCandidates64_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail65
    (if 3 ≤ smallCandidates64.length then search relation 3 [3, 3119] smallCandidates64 3122 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan65

def smallTail63 : List Nat := 3049 :: smallTail64
def smallCandidates63 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates63_checked : candidates relation 4 3 3049 smallTail64 smallBest.bound = smallCandidates63 := by decide

theorem smallScan63 : scan relation (search relation 3) 4 [3] 3 smallTail63 smallBest = smallBest := by
  rw [smallTail63, scan, if_neg (by decide), smallCandidates63_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail64
    (if 3 ≤ smallCandidates63.length then search relation 3 [3, 3049] smallCandidates63 3052 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan64

def smallTail62 : List Nat := 2957 :: smallTail63
def smallCandidates62 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates62_checked : candidates relation 4 3 2957 smallTail63 smallBest.bound = smallCandidates62 := by decide

theorem smallScan62 : scan relation (search relation 3) 4 [3] 3 smallTail62 smallBest = smallBest := by
  rw [smallTail62, scan, if_neg (by decide), smallCandidates62_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail63
    (if 3 ≤ smallCandidates62.length then search relation 3 [3, 2957] smallCandidates62 2960 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan63

def smallTail61 : List Nat := 2917 :: smallTail62
def smallCandidates61 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates61_checked : candidates relation 4 3 2917 smallTail62 smallBest.bound = smallCandidates61 := by decide

theorem smallScan61 : scan relation (search relation 3) 4 [3] 3 smallTail61 smallBest = smallBest := by
  rw [smallTail61, scan, if_neg (by decide), smallCandidates61_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail62
    (if 3 ≤ smallCandidates61.length then search relation 3 [3, 2917] smallCandidates61 2920 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan62

end ProjectEulerSolutions.PrimePairSearch
