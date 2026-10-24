import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates540 : List Nat := [4297, 4567, 4591, 4957, 5179, 5281, 5503, 5743, 6007, 6151, 6397, 6529, 6961, 7309, 7489, 7789, 8011, 8317]
set_option maxRecDepth 4096 in
theorem rootCandidates540_checked : candidates edge 5 0 3919 tail541 best.bound = rootCandidates540 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext q; rfl
  rw [row]
  decide

def innerTail540_18 : List Nat := []
theorem innerScan540_18 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_18 best = best := rfl

def innerTail540_17 : List Nat := 8317 :: innerTail540_18
def innerCandidates540_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates540_17_checked : candidates edge 4 3919 8317 innerTail540_18 best.bound = innerCandidates540_17 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_17 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_17 best = best := by
  rw [innerTail540_17, scan, if_neg (by decide), innerCandidates540_17_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_18
    (if 3 ≤ innerCandidates540_17.length then search edge 3 [3919, 8317] innerCandidates540_17 12236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_18

def innerTail540_16 : List Nat := 8011 :: innerTail540_17
def innerCandidates540_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates540_16_checked : candidates edge 4 3919 8011 innerTail540_17 best.bound = innerCandidates540_16 := by
  unfold candidates
  have row : edge 8011 = SearchData.row8011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_16 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_16 best = best := by
  rw [innerTail540_16, scan, if_neg (by decide), innerCandidates540_16_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_17
    (if 3 ≤ innerCandidates540_16.length then search edge 3 [3919, 8011] innerCandidates540_16 11930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_17

def innerTail540_15 : List Nat := 7789 :: innerTail540_16
def innerCandidates540_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates540_15_checked : candidates edge 4 3919 7789 innerTail540_16 best.bound = innerCandidates540_15 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_15 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_15 best = best := by
  rw [innerTail540_15, scan, if_neg (by decide), innerCandidates540_15_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_16
    (if 3 ≤ innerCandidates540_15.length then search edge 3 [3919, 7789] innerCandidates540_15 11708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_16

def innerTail540_14 : List Nat := 7489 :: innerTail540_15
def innerCandidates540_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates540_14_checked : candidates edge 4 3919 7489 innerTail540_15 best.bound = innerCandidates540_14 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_14 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_14 best = best := by
  rw [innerTail540_14, scan, if_neg (by decide), innerCandidates540_14_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_15
    (if 3 ≤ innerCandidates540_14.length then search edge 3 [3919, 7489] innerCandidates540_14 11408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_15

def innerTail540_13 : List Nat := 7309 :: innerTail540_14
def innerCandidates540_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates540_13_checked : candidates edge 4 3919 7309 innerTail540_14 best.bound = innerCandidates540_13 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_13 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_13 best = best := by
  rw [innerTail540_13, scan, if_neg (by decide), innerCandidates540_13_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_14
    (if 3 ≤ innerCandidates540_13.length then search edge 3 [3919, 7309] innerCandidates540_13 11228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_14

def innerTail540_12 : List Nat := 6961 :: innerTail540_13
def innerCandidates540_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates540_12_checked : candidates edge 4 3919 6961 innerTail540_13 best.bound = innerCandidates540_12 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_12 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_12 best = best := by
  rw [innerTail540_12, scan, if_neg (by decide), innerCandidates540_12_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_13
    (if 3 ≤ innerCandidates540_12.length then search edge 3 [3919, 6961] innerCandidates540_12 10880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_13

def innerTail540_11 : List Nat := 6529 :: innerTail540_12
def innerCandidates540_11 : List Nat := [6961]
set_option maxRecDepth 4096 in
theorem innerCandidates540_11_checked : candidates edge 4 3919 6529 innerTail540_12 best.bound = innerCandidates540_11 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_11 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_11 best = best := by
  rw [innerTail540_11, scan, if_neg (by decide), innerCandidates540_11_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_12
    (if 3 ≤ innerCandidates540_11.length then search edge 3 [3919, 6529] innerCandidates540_11 10448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_12

def innerTail540_10 : List Nat := 6397 :: innerTail540_11
def innerCandidates540_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates540_10_checked : candidates edge 4 3919 6397 innerTail540_11 best.bound = innerCandidates540_10 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_10 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_10 best = best := by
  rw [innerTail540_10, scan, if_neg (by decide), innerCandidates540_10_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_11
    (if 3 ≤ innerCandidates540_10.length then search edge 3 [3919, 6397] innerCandidates540_10 10316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_11

def innerTail540_9 : List Nat := 6151 :: innerTail540_10
def innerCandidates540_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates540_9_checked : candidates edge 4 3919 6151 innerTail540_10 best.bound = innerCandidates540_9 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_9 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_9 best = best := by
  rw [innerTail540_9, scan, if_neg (by decide), innerCandidates540_9_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_10
    (if 3 ≤ innerCandidates540_9.length then search edge 3 [3919, 6151] innerCandidates540_9 10070 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_10

def innerTail540_8 : List Nat := 6007 :: innerTail540_9
def innerCandidates540_8 : List Nat := [7489]
set_option maxRecDepth 4096 in
theorem innerCandidates540_8_checked : candidates edge 4 3919 6007 innerTail540_9 best.bound = innerCandidates540_8 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_8 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_8 best = best := by
  rw [innerTail540_8, scan, if_neg (by decide), innerCandidates540_8_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_9
    (if 3 ≤ innerCandidates540_8.length then search edge 3 [3919, 6007] innerCandidates540_8 9926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_9

def innerTail540_7 : List Nat := 5743 :: innerTail540_8
def innerCandidates540_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates540_7_checked : candidates edge 4 3919 5743 innerTail540_8 best.bound = innerCandidates540_7 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_7 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_7 best = best := by
  rw [innerTail540_7, scan, if_neg (by decide), innerCandidates540_7_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_8
    (if 3 ≤ innerCandidates540_7.length then search edge 3 [3919, 5743] innerCandidates540_7 9662 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_8

def innerTail540_6 : List Nat := 5503 :: innerTail540_7
def innerCandidates540_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates540_6_checked : candidates edge 4 3919 5503 innerTail540_7 best.bound = innerCandidates540_6 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_6 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_6 best = best := by
  rw [innerTail540_6, scan, if_neg (by decide), innerCandidates540_6_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_7
    (if 3 ≤ innerCandidates540_6.length then search edge 3 [3919, 5503] innerCandidates540_6 9422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_7

def innerTail540_5 : List Nat := 5281 :: innerTail540_6
def innerCandidates540_5 : List Nat := [7489]
set_option maxRecDepth 4096 in
theorem innerCandidates540_5_checked : candidates edge 4 3919 5281 innerTail540_6 best.bound = innerCandidates540_5 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_5 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_5 best = best := by
  rw [innerTail540_5, scan, if_neg (by decide), innerCandidates540_5_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_6
    (if 3 ≤ innerCandidates540_5.length then search edge 3 [3919, 5281] innerCandidates540_5 9200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_6

def innerTail540_4 : List Nat := 5179 :: innerTail540_5
def innerCandidates540_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates540_4_checked : candidates edge 4 3919 5179 innerTail540_5 best.bound = innerCandidates540_4 := by
  unfold candidates
  have row : edge 5179 = SearchData.row5179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_4 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_4 best = best := by
  rw [innerTail540_4, scan, if_neg (by decide), innerCandidates540_4_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_5
    (if 3 ≤ innerCandidates540_4.length then search edge 3 [3919, 5179] innerCandidates540_4 9098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_5

def innerTail540_3 : List Nat := 4957 :: innerTail540_4
def innerCandidates540_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates540_3_checked : candidates edge 4 3919 4957 innerTail540_4 best.bound = innerCandidates540_3 := by
  unfold candidates
  have row : edge 4957 = SearchData.row4957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_3 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_3 best = best := by
  rw [innerTail540_3, scan, if_neg (by decide), innerCandidates540_3_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_4
    (if 3 ≤ innerCandidates540_3.length then search edge 3 [3919, 4957] innerCandidates540_3 8876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_4

def innerTail540_2 : List Nat := 4591 :: innerTail540_3
def innerCandidates540_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates540_2_checked : candidates edge 4 3919 4591 innerTail540_3 best.bound = innerCandidates540_2 := by
  unfold candidates
  have row : edge 4591 = SearchData.row4591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_2 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_2 best = best := by
  rw [innerTail540_2, scan, if_neg (by decide), innerCandidates540_2_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_3
    (if 3 ≤ innerCandidates540_2.length then search edge 3 [3919, 4591] innerCandidates540_2 8510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_3

def innerTail540_1 : List Nat := 4567 :: innerTail540_2
def innerCandidates540_1 : List Nat := [7789]
set_option maxRecDepth 4096 in
theorem innerCandidates540_1_checked : candidates edge 4 3919 4567 innerTail540_2 best.bound = innerCandidates540_1 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_1 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_1 best = best := by
  rw [innerTail540_1, scan, if_neg (by decide), innerCandidates540_1_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_2
    (if 3 ≤ innerCandidates540_1.length then search edge 3 [3919, 4567] innerCandidates540_1 8486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_2

def innerTail540_0 : List Nat := 4297 :: innerTail540_1
def innerCandidates540_0 : List Nat := [8317]
set_option maxRecDepth 4096 in
theorem innerCandidates540_0_checked : candidates edge 4 3919 4297 innerTail540_1 best.bound = innerCandidates540_0 := by
  unfold candidates
  have row : edge 4297 = SearchData.row4297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan540_0 : scan edge (search edge 3) 4 [3919] 3919 innerTail540_0 best = best := by
  rw [innerTail540_0, scan, if_neg (by decide), innerCandidates540_0_checked]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_1
    (if 3 ≤ innerCandidates540_0.length then search edge 3 [3919, 4297] innerCandidates540_0 8216 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan540_1

theorem rootBranch540 : search edge 4 [3919] rootCandidates540 3919 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3919] 3919 innerTail540_0 best = best
  exact innerScan540_0

def rootCandidates539 : List Nat := [4019, 4211, 4271, 4337, 4349, 4649, 5189, 5399, 5483, 5987, 6323, 6779, 6899, 7331, 7499]
set_option maxRecDepth 4096 in
theorem rootCandidates539_checked : candidates edge 5 0 3917 tail540 best.bound = rootCandidates539 := by
  unfold candidates
  have row : edge 3917 = SearchData.row3917.contains := by funext q; rfl
  rw [row]
  decide

def innerTail539_15 : List Nat := []
theorem innerScan539_15 : scan edge (search edge 3) 4 [3917] 3917 innerTail539_15 best = best := rfl

def innerTail539_14 : List Nat := 7499 :: innerTail539_15
def innerCandidates539_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates539_14_checked : candidates edge 4 3917 7499 innerTail539_15 best.bound = innerCandidates539_14 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan539_14 : scan edge (search edge 3) 4 [3917] 3917 innerTail539_14 best = best := by
  rw [innerTail539_14, scan, if_neg (by decide), innerCandidates539_14_checked]
  change scan edge (search edge 3) 4 [3917] 3917 innerTail539_15
    (if 3 ≤ innerCandidates539_14.length then search edge 3 [3917, 7499] innerCandidates539_14 11416 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan539_15

def innerTail539_13 : List Nat := 7331 :: innerTail539_14
def innerCandidates539_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates539_13_checked : candidates edge 4 3917 7331 innerTail539_14 best.bound = innerCandidates539_13 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan539_13 : scan edge (search edge 3) 4 [3917] 3917 innerTail539_13 best = best := by
  rw [innerTail539_13, scan, if_neg (by decide), innerCandidates539_13_checked]
  change scan edge (search edge 3) 4 [3917] 3917 innerTail539_14
    (if 3 ≤ innerCandidates539_13.length then search edge 3 [3917, 7331] innerCandidates539_13 11248 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan539_14

def innerTail539_12 : List Nat := 6899 :: innerTail539_13
def innerCandidates539_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates539_12_checked : candidates edge 4 3917 6899 innerTail539_13 best.bound = innerCandidates539_12 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan539_12 : scan edge (search edge 3) 4 [3917] 3917 innerTail539_12 best = best := by
  rw [innerTail539_12, scan, if_neg (by decide), innerCandidates539_12_checked]
  change scan edge (search edge 3) 4 [3917] 3917 innerTail539_13
    (if 3 ≤ innerCandidates539_12.length then search edge 3 [3917, 6899] innerCandidates539_12 10816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan539_13

def innerTail539_11 : List Nat := 6779 :: innerTail539_12
def innerCandidates539_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates539_11_checked : candidates edge 4 3917 6779 innerTail539_12 best.bound = innerCandidates539_11 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan539_11 : scan edge (search edge 3) 4 [3917] 3917 innerTail539_11 best = best := by
  rw [innerTail539_11, scan, if_neg (by decide), innerCandidates539_11_checked]
  change scan edge (search edge 3) 4 [3917] 3917 innerTail539_12
    (if 3 ≤ innerCandidates539_11.length then search edge 3 [3917, 6779] innerCandidates539_11 10696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan539_12

def innerTail539_10 : List Nat := 6323 :: innerTail539_11
def innerCandidates539_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates539_10_checked : candidates edge 4 3917 6323 innerTail539_11 best.bound = innerCandidates539_10 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan539_10 : scan edge (search edge 3) 4 [3917] 3917 innerTail539_10 best = best := by
  rw [innerTail539_10, scan, if_neg (by decide), innerCandidates539_10_checked]
  change scan edge (search edge 3) 4 [3917] 3917 innerTail539_11
    (if 3 ≤ innerCandidates539_10.length then search edge 3 [3917, 6323] innerCandidates539_10 10240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan539_11

def innerTail539_9 : List Nat := 5987 :: innerTail539_10
def innerCandidates539_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates539_9_checked : candidates edge 4 3917 5987 innerTail539_10 best.bound = innerCandidates539_9 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan539_9 : scan edge (search edge 3) 4 [3917] 3917 innerTail539_9 best = best := by
  rw [innerTail539_9, scan, if_neg (by decide), innerCandidates539_9_checked]
  change scan edge (search edge 3) 4 [3917] 3917 innerTail539_10
    (if 3 ≤ innerCandidates539_9.length then search edge 3 [3917, 5987] innerCandidates539_9 9904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan539_10

def innerTail539_8 : List Nat := 5483 :: innerTail539_9
def innerCandidates539_8 : List Nat := [5987]
set_option maxRecDepth 4096 in
theorem innerCandidates539_8_checked : candidates edge 4 3917 5483 innerTail539_9 best.bound = innerCandidates539_8 := by
  unfold candidates
  have row : edge 5483 = SearchData.row5483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan539_8 : scan edge (search edge 3) 4 [3917] 3917 innerTail539_8 best = best := by
  rw [innerTail539_8, scan, if_neg (by decide), innerCandidates539_8_checked]
  change scan edge (search edge 3) 4 [3917] 3917 innerTail539_9
    (if 3 ≤ innerCandidates539_8.length then search edge 3 [3917, 5483] innerCandidates539_8 9400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan539_9

def innerTail539_7 : List Nat := 5399 :: innerTail539_8
def innerCandidates539_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates539_7_checked : candidates edge 4 3917 5399 innerTail539_8 best.bound = innerCandidates539_7 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan539_7 : scan edge (search edge 3) 4 [3917] 3917 innerTail539_7 best = best := by
  rw [innerTail539_7, scan, if_neg (by decide), innerCandidates539_7_checked]
  change scan edge (search edge 3) 4 [3917] 3917 innerTail539_8
    (if 3 ≤ innerCandidates539_7.length then search edge 3 [3917, 5399] innerCandidates539_7 9316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan539_8

def innerTail539_6 : List Nat := 5189 :: innerTail539_7
def innerCandidates539_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates539_6_checked : candidates edge 4 3917 5189 innerTail539_7 best.bound = innerCandidates539_6 := by
  unfold candidates
  have row : edge 5189 = SearchData.row5189.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan539_6 : scan edge (search edge 3) 4 [3917] 3917 innerTail539_6 best = best := by
  rw [innerTail539_6, scan, if_neg (by decide), innerCandidates539_6_checked]
  change scan edge (search edge 3) 4 [3917] 3917 innerTail539_7
    (if 3 ≤ innerCandidates539_6.length then search edge 3 [3917, 5189] innerCandidates539_6 9106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan539_7

def innerTail539_5 : List Nat := 4649 :: innerTail539_6
def innerCandidates539_5 : List Nat := [5399, 5987, 6323]
set_option maxRecDepth 4096 in
theorem innerCandidates539_5_checked : candidates edge 4 3917 4649 innerTail539_6 best.bound = innerCandidates539_5 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild539_5 : search edge 3 [3917, 4649] innerCandidates539_5 8566 best = best := by decide

theorem innerScan539_5 : scan edge (search edge 3) 4 [3917] 3917 innerTail539_5 best = best := by
  rw [innerTail539_5, scan, if_neg (by decide), innerCandidates539_5_checked]
  change scan edge (search edge 3) 4 [3917] 3917 innerTail539_6
    (if 3 ≤ innerCandidates539_5.length then search edge 3 [3917, 4649] innerCandidates539_5 8566 best else best) = best
  rw [if_pos (by decide), innerChild539_5]
  exact innerScan539_6

def innerTail539_4 : List Nat := 4349 :: innerTail539_5
def innerCandidates539_4 : List Nat := [5483]
set_option maxRecDepth 4096 in
theorem innerCandidates539_4_checked : candidates edge 4 3917 4349 innerTail539_5 best.bound = innerCandidates539_4 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan539_4 : scan edge (search edge 3) 4 [3917] 3917 innerTail539_4 best = best := by
  rw [innerTail539_4, scan, if_neg (by decide), innerCandidates539_4_checked]
  change scan edge (search edge 3) 4 [3917] 3917 innerTail539_5
    (if 3 ≤ innerCandidates539_4.length then search edge 3 [3917, 4349] innerCandidates539_4 8266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan539_5

def innerTail539_3 : List Nat := 4337 :: innerTail539_4
def innerCandidates539_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates539_3_checked : candidates edge 4 3917 4337 innerTail539_4 best.bound = innerCandidates539_3 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan539_3 : scan edge (search edge 3) 4 [3917] 3917 innerTail539_3 best = best := by
  rw [innerTail539_3, scan, if_neg (by decide), innerCandidates539_3_checked]
  change scan edge (search edge 3) 4 [3917] 3917 innerTail539_4
    (if 3 ≤ innerCandidates539_3.length then search edge 3 [3917, 4337] innerCandidates539_3 8254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan539_4

def innerTail539_2 : List Nat := 4271 :: innerTail539_3
def innerCandidates539_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates539_2_checked : candidates edge 4 3917 4271 innerTail539_3 best.bound = innerCandidates539_2 := by
  unfold candidates
  have row : edge 4271 = SearchData.row4271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan539_2 : scan edge (search edge 3) 4 [3917] 3917 innerTail539_2 best = best := by
  rw [innerTail539_2, scan, if_neg (by decide), innerCandidates539_2_checked]
  change scan edge (search edge 3) 4 [3917] 3917 innerTail539_3
    (if 3 ≤ innerCandidates539_2.length then search edge 3 [3917, 4271] innerCandidates539_2 8188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan539_3

def innerTail539_1 : List Nat := 4211 :: innerTail539_2
def innerCandidates539_1 : List Nat := [5483]
set_option maxRecDepth 4096 in
theorem innerCandidates539_1_checked : candidates edge 4 3917 4211 innerTail539_2 best.bound = innerCandidates539_1 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan539_1 : scan edge (search edge 3) 4 [3917] 3917 innerTail539_1 best = best := by
  rw [innerTail539_1, scan, if_neg (by decide), innerCandidates539_1_checked]
  change scan edge (search edge 3) 4 [3917] 3917 innerTail539_2
    (if 3 ≤ innerCandidates539_1.length then search edge 3 [3917, 4211] innerCandidates539_1 8128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan539_2

def innerTail539_0 : List Nat := 4019 :: innerTail539_1
def innerCandidates539_0 : List Nat := [4211]
set_option maxRecDepth 4096 in
theorem innerCandidates539_0_checked : candidates edge 4 3917 4019 innerTail539_1 best.bound = innerCandidates539_0 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan539_0 : scan edge (search edge 3) 4 [3917] 3917 innerTail539_0 best = best := by
  rw [innerTail539_0, scan, if_neg (by decide), innerCandidates539_0_checked]
  change scan edge (search edge 3) 4 [3917] 3917 innerTail539_1
    (if 3 ≤ innerCandidates539_0.length then search edge 3 [3917, 4019] innerCandidates539_0 7936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan539_1

theorem rootBranch539 : search edge 4 [3917] rootCandidates539 3917 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3917] 3917 innerTail539_0 best = best
  exact innerScan539_0

def rootCandidates538 : List Nat := [4079, 4259, 4457, 5477, 5717, 6257, 7349, 7919, 7937]
set_option maxRecDepth 4096 in
theorem rootCandidates538_checked : candidates edge 5 0 3911 tail539 best.bound = rootCandidates538 := by
  unfold candidates
  have row : edge 3911 = SearchData.row3911.contains := by funext q; rfl
  rw [row]
  decide

def innerTail538_9 : List Nat := []
theorem innerScan538_9 : scan edge (search edge 3) 4 [3911] 3911 innerTail538_9 best = best := rfl

def innerTail538_8 : List Nat := 7937 :: innerTail538_9
def innerCandidates538_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates538_8_checked : candidates edge 4 3911 7937 innerTail538_9 best.bound = innerCandidates538_8 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan538_8 : scan edge (search edge 3) 4 [3911] 3911 innerTail538_8 best = best := by
  rw [innerTail538_8, scan, if_neg (by decide), innerCandidates538_8_checked]
  change scan edge (search edge 3) 4 [3911] 3911 innerTail538_9
    (if 3 ≤ innerCandidates538_8.length then search edge 3 [3911, 7937] innerCandidates538_8 11848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan538_9

def innerTail538_7 : List Nat := 7919 :: innerTail538_8
def innerCandidates538_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates538_7_checked : candidates edge 4 3911 7919 innerTail538_8 best.bound = innerCandidates538_7 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan538_7 : scan edge (search edge 3) 4 [3911] 3911 innerTail538_7 best = best := by
  rw [innerTail538_7, scan, if_neg (by decide), innerCandidates538_7_checked]
  change scan edge (search edge 3) 4 [3911] 3911 innerTail538_8
    (if 3 ≤ innerCandidates538_7.length then search edge 3 [3911, 7919] innerCandidates538_7 11830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan538_8

def innerTail538_6 : List Nat := 7349 :: innerTail538_7
def innerCandidates538_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates538_6_checked : candidates edge 4 3911 7349 innerTail538_7 best.bound = innerCandidates538_6 := by
  unfold candidates
  have row : edge 7349 = SearchData.row7349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan538_6 : scan edge (search edge 3) 4 [3911] 3911 innerTail538_6 best = best := by
  rw [innerTail538_6, scan, if_neg (by decide), innerCandidates538_6_checked]
  change scan edge (search edge 3) 4 [3911] 3911 innerTail538_7
    (if 3 ≤ innerCandidates538_6.length then search edge 3 [3911, 7349] innerCandidates538_6 11260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan538_7

def innerTail538_5 : List Nat := 6257 :: innerTail538_6
def innerCandidates538_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates538_5_checked : candidates edge 4 3911 6257 innerTail538_6 best.bound = innerCandidates538_5 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan538_5 : scan edge (search edge 3) 4 [3911] 3911 innerTail538_5 best = best := by
  rw [innerTail538_5, scan, if_neg (by decide), innerCandidates538_5_checked]
  change scan edge (search edge 3) 4 [3911] 3911 innerTail538_6
    (if 3 ≤ innerCandidates538_5.length then search edge 3 [3911, 6257] innerCandidates538_5 10168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan538_6

def innerTail538_4 : List Nat := 5717 :: innerTail538_5
def innerCandidates538_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates538_4_checked : candidates edge 4 3911 5717 innerTail538_5 best.bound = innerCandidates538_4 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan538_4 : scan edge (search edge 3) 4 [3911] 3911 innerTail538_4 best = best := by
  rw [innerTail538_4, scan, if_neg (by decide), innerCandidates538_4_checked]
  change scan edge (search edge 3) 4 [3911] 3911 innerTail538_5
    (if 3 ≤ innerCandidates538_4.length then search edge 3 [3911, 5717] innerCandidates538_4 9628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan538_5

def innerTail538_3 : List Nat := 5477 :: innerTail538_4
def innerCandidates538_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates538_3_checked : candidates edge 4 3911 5477 innerTail538_4 best.bound = innerCandidates538_3 := by
  unfold candidates
  have row : edge 5477 = SearchData.row5477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan538_3 : scan edge (search edge 3) 4 [3911] 3911 innerTail538_3 best = best := by
  rw [innerTail538_3, scan, if_neg (by decide), innerCandidates538_3_checked]
  change scan edge (search edge 3) 4 [3911] 3911 innerTail538_4
    (if 3 ≤ innerCandidates538_3.length then search edge 3 [3911, 5477] innerCandidates538_3 9388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan538_4

def innerTail538_2 : List Nat := 4457 :: innerTail538_3
def innerCandidates538_2 : List Nat := [7919]
set_option maxRecDepth 4096 in
theorem innerCandidates538_2_checked : candidates edge 4 3911 4457 innerTail538_3 best.bound = innerCandidates538_2 := by
  unfold candidates
  have row : edge 4457 = SearchData.row4457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan538_2 : scan edge (search edge 3) 4 [3911] 3911 innerTail538_2 best = best := by
  rw [innerTail538_2, scan, if_neg (by decide), innerCandidates538_2_checked]
  change scan edge (search edge 3) 4 [3911] 3911 innerTail538_3
    (if 3 ≤ innerCandidates538_2.length then search edge 3 [3911, 4457] innerCandidates538_2 8368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan538_3

def innerTail538_1 : List Nat := 4259 :: innerTail538_2
def innerCandidates538_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates538_1_checked : candidates edge 4 3911 4259 innerTail538_2 best.bound = innerCandidates538_1 := by
  unfold candidates
  have row : edge 4259 = SearchData.row4259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan538_1 : scan edge (search edge 3) 4 [3911] 3911 innerTail538_1 best = best := by
  rw [innerTail538_1, scan, if_neg (by decide), innerCandidates538_1_checked]
  change scan edge (search edge 3) 4 [3911] 3911 innerTail538_2
    (if 3 ≤ innerCandidates538_1.length then search edge 3 [3911, 4259] innerCandidates538_1 8170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan538_2

def innerTail538_0 : List Nat := 4079 :: innerTail538_1
def innerCandidates538_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates538_0_checked : candidates edge 4 3911 4079 innerTail538_1 best.bound = innerCandidates538_0 := by
  unfold candidates
  have row : edge 4079 = SearchData.row4079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan538_0 : scan edge (search edge 3) 4 [3911] 3911 innerTail538_0 best = best := by
  rw [innerTail538_0, scan, if_neg (by decide), innerCandidates538_0_checked]
  change scan edge (search edge 3) 4 [3911] 3911 innerTail538_1
    (if 3 ≤ innerCandidates538_0.length then search edge 3 [3911, 4079] innerCandidates538_0 7990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan538_1

theorem rootBranch538 : search edge 4 [3911] rootCandidates538 3911 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3911] 3911 innerTail538_0 best = best
  exact innerScan538_0

def rootCandidates537 : List Nat := [4027, 4327, 4603, 5431, 6043, 6469, 6871, 6997, 7333, 7573]
set_option maxRecDepth 4096 in
theorem rootCandidates537_checked : candidates edge 5 0 3907 tail538 best.bound = rootCandidates537 := by
  unfold candidates
  have row : edge 3907 = SearchData.row3907.contains := by funext q; rfl
  rw [row]
  decide

def innerTail537_10 : List Nat := []
theorem innerScan537_10 : scan edge (search edge 3) 4 [3907] 3907 innerTail537_10 best = best := rfl

def innerTail537_9 : List Nat := 7573 :: innerTail537_10
def innerCandidates537_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates537_9_checked : candidates edge 4 3907 7573 innerTail537_10 best.bound = innerCandidates537_9 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan537_9 : scan edge (search edge 3) 4 [3907] 3907 innerTail537_9 best = best := by
  rw [innerTail537_9, scan, if_neg (by decide), innerCandidates537_9_checked]
  change scan edge (search edge 3) 4 [3907] 3907 innerTail537_10
    (if 3 ≤ innerCandidates537_9.length then search edge 3 [3907, 7573] innerCandidates537_9 11480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan537_10

def innerTail537_8 : List Nat := 7333 :: innerTail537_9
def innerCandidates537_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates537_8_checked : candidates edge 4 3907 7333 innerTail537_9 best.bound = innerCandidates537_8 := by
  unfold candidates
  have row : edge 7333 = SearchData.row7333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan537_8 : scan edge (search edge 3) 4 [3907] 3907 innerTail537_8 best = best := by
  rw [innerTail537_8, scan, if_neg (by decide), innerCandidates537_8_checked]
  change scan edge (search edge 3) 4 [3907] 3907 innerTail537_9
    (if 3 ≤ innerCandidates537_8.length then search edge 3 [3907, 7333] innerCandidates537_8 11240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan537_9

def innerTail537_7 : List Nat := 6997 :: innerTail537_8
def innerCandidates537_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates537_7_checked : candidates edge 4 3907 6997 innerTail537_8 best.bound = innerCandidates537_7 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan537_7 : scan edge (search edge 3) 4 [3907] 3907 innerTail537_7 best = best := by
  rw [innerTail537_7, scan, if_neg (by decide), innerCandidates537_7_checked]
  change scan edge (search edge 3) 4 [3907] 3907 innerTail537_8
    (if 3 ≤ innerCandidates537_7.length then search edge 3 [3907, 6997] innerCandidates537_7 10904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan537_8

def innerTail537_6 : List Nat := 6871 :: innerTail537_7
def innerCandidates537_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates537_6_checked : candidates edge 4 3907 6871 innerTail537_7 best.bound = innerCandidates537_6 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan537_6 : scan edge (search edge 3) 4 [3907] 3907 innerTail537_6 best = best := by
  rw [innerTail537_6, scan, if_neg (by decide), innerCandidates537_6_checked]
  change scan edge (search edge 3) 4 [3907] 3907 innerTail537_7
    (if 3 ≤ innerCandidates537_6.length then search edge 3 [3907, 6871] innerCandidates537_6 10778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan537_7

def innerTail537_5 : List Nat := 6469 :: innerTail537_6
def innerCandidates537_5 : List Nat := [6997]
set_option maxRecDepth 4096 in
theorem innerCandidates537_5_checked : candidates edge 4 3907 6469 innerTail537_6 best.bound = innerCandidates537_5 := by
  unfold candidates
  have row : edge 6469 = SearchData.row6469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan537_5 : scan edge (search edge 3) 4 [3907] 3907 innerTail537_5 best = best := by
  rw [innerTail537_5, scan, if_neg (by decide), innerCandidates537_5_checked]
  change scan edge (search edge 3) 4 [3907] 3907 innerTail537_6
    (if 3 ≤ innerCandidates537_5.length then search edge 3 [3907, 6469] innerCandidates537_5 10376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan537_6

def innerTail537_4 : List Nat := 6043 :: innerTail537_5
def innerCandidates537_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates537_4_checked : candidates edge 4 3907 6043 innerTail537_5 best.bound = innerCandidates537_4 := by
  unfold candidates
  have row : edge 6043 = SearchData.row6043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan537_4 : scan edge (search edge 3) 4 [3907] 3907 innerTail537_4 best = best := by
  rw [innerTail537_4, scan, if_neg (by decide), innerCandidates537_4_checked]
  change scan edge (search edge 3) 4 [3907] 3907 innerTail537_5
    (if 3 ≤ innerCandidates537_4.length then search edge 3 [3907, 6043] innerCandidates537_4 9950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan537_5

def innerTail537_3 : List Nat := 5431 :: innerTail537_4
def innerCandidates537_3 : List Nat := [7573]
set_option maxRecDepth 4096 in
theorem innerCandidates537_3_checked : candidates edge 4 3907 5431 innerTail537_4 best.bound = innerCandidates537_3 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan537_3 : scan edge (search edge 3) 4 [3907] 3907 innerTail537_3 best = best := by
  rw [innerTail537_3, scan, if_neg (by decide), innerCandidates537_3_checked]
  change scan edge (search edge 3) 4 [3907] 3907 innerTail537_4
    (if 3 ≤ innerCandidates537_3.length then search edge 3 [3907, 5431] innerCandidates537_3 9338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan537_4

def innerTail537_2 : List Nat := 4603 :: innerTail537_3
def innerCandidates537_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates537_2_checked : candidates edge 4 3907 4603 innerTail537_3 best.bound = innerCandidates537_2 := by
  unfold candidates
  have row : edge 4603 = SearchData.row4603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan537_2 : scan edge (search edge 3) 4 [3907] 3907 innerTail537_2 best = best := by
  rw [innerTail537_2, scan, if_neg (by decide), innerCandidates537_2_checked]
  change scan edge (search edge 3) 4 [3907] 3907 innerTail537_3
    (if 3 ≤ innerCandidates537_2.length then search edge 3 [3907, 4603] innerCandidates537_2 8510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan537_3

def innerTail537_1 : List Nat := 4327 :: innerTail537_2
def innerCandidates537_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates537_1_checked : candidates edge 4 3907 4327 innerTail537_2 best.bound = innerCandidates537_1 := by
  unfold candidates
  have row : edge 4327 = SearchData.row4327.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan537_1 : scan edge (search edge 3) 4 [3907] 3907 innerTail537_1 best = best := by
  rw [innerTail537_1, scan, if_neg (by decide), innerCandidates537_1_checked]
  change scan edge (search edge 3) 4 [3907] 3907 innerTail537_2
    (if 3 ≤ innerCandidates537_1.length then search edge 3 [3907, 4327] innerCandidates537_1 8234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan537_2

def innerTail537_0 : List Nat := 4027 :: innerTail537_1
def innerCandidates537_0 : List Nat := [6871]
set_option maxRecDepth 4096 in
theorem innerCandidates537_0_checked : candidates edge 4 3907 4027 innerTail537_1 best.bound = innerCandidates537_0 := by
  unfold candidates
  have row : edge 4027 = SearchData.row4027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan537_0 : scan edge (search edge 3) 4 [3907] 3907 innerTail537_0 best = best := by
  rw [innerTail537_0, scan, if_neg (by decide), innerCandidates537_0_checked]
  change scan edge (search edge 3) 4 [3907] 3907 innerTail537_1
    (if 3 ≤ innerCandidates537_0.length then search edge 3 [3907, 4027] innerCandidates537_0 7934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan537_1

theorem rootBranch537 : search edge 4 [3907] rootCandidates537 3907 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3907] 3907 innerTail537_0 best = best
  exact innerScan537_0

def rootCandidates536 : List Nat := [4021, 5749, 5821, 6007, 6619, 6763, 7477, 7927]
set_option maxRecDepth 4096 in
theorem rootCandidates536_checked : candidates edge 5 0 3889 tail537 best.bound = rootCandidates536 := by
  unfold candidates
  have row : edge 3889 = SearchData.row3889.contains := by funext q; rfl
  rw [row]
  decide

def innerTail536_8 : List Nat := []
theorem innerScan536_8 : scan edge (search edge 3) 4 [3889] 3889 innerTail536_8 best = best := rfl

def innerTail536_7 : List Nat := 7927 :: innerTail536_8
def innerCandidates536_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates536_7_checked : candidates edge 4 3889 7927 innerTail536_8 best.bound = innerCandidates536_7 := by
  unfold candidates
  have row : edge 7927 = SearchData.row7927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan536_7 : scan edge (search edge 3) 4 [3889] 3889 innerTail536_7 best = best := by
  rw [innerTail536_7, scan, if_neg (by decide), innerCandidates536_7_checked]
  change scan edge (search edge 3) 4 [3889] 3889 innerTail536_8
    (if 3 ≤ innerCandidates536_7.length then search edge 3 [3889, 7927] innerCandidates536_7 11816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan536_8

def innerTail536_6 : List Nat := 7477 :: innerTail536_7
def innerCandidates536_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates536_6_checked : candidates edge 4 3889 7477 innerTail536_7 best.bound = innerCandidates536_6 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan536_6 : scan edge (search edge 3) 4 [3889] 3889 innerTail536_6 best = best := by
  rw [innerTail536_6, scan, if_neg (by decide), innerCandidates536_6_checked]
  change scan edge (search edge 3) 4 [3889] 3889 innerTail536_7
    (if 3 ≤ innerCandidates536_6.length then search edge 3 [3889, 7477] innerCandidates536_6 11366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan536_7

def innerTail536_5 : List Nat := 6763 :: innerTail536_6
def innerCandidates536_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates536_5_checked : candidates edge 4 3889 6763 innerTail536_6 best.bound = innerCandidates536_5 := by
  unfold candidates
  have row : edge 6763 = SearchData.row6763.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan536_5 : scan edge (search edge 3) 4 [3889] 3889 innerTail536_5 best = best := by
  rw [innerTail536_5, scan, if_neg (by decide), innerCandidates536_5_checked]
  change scan edge (search edge 3) 4 [3889] 3889 innerTail536_6
    (if 3 ≤ innerCandidates536_5.length then search edge 3 [3889, 6763] innerCandidates536_5 10652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan536_6

def innerTail536_4 : List Nat := 6619 :: innerTail536_5
def innerCandidates536_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates536_4_checked : candidates edge 4 3889 6619 innerTail536_5 best.bound = innerCandidates536_4 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan536_4 : scan edge (search edge 3) 4 [3889] 3889 innerTail536_4 best = best := by
  rw [innerTail536_4, scan, if_neg (by decide), innerCandidates536_4_checked]
  change scan edge (search edge 3) 4 [3889] 3889 innerTail536_5
    (if 3 ≤ innerCandidates536_4.length then search edge 3 [3889, 6619] innerCandidates536_4 10508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan536_5

def innerTail536_3 : List Nat := 6007 :: innerTail536_4
def innerCandidates536_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates536_3_checked : candidates edge 4 3889 6007 innerTail536_4 best.bound = innerCandidates536_3 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan536_3 : scan edge (search edge 3) 4 [3889] 3889 innerTail536_3 best = best := by
  rw [innerTail536_3, scan, if_neg (by decide), innerCandidates536_3_checked]
  change scan edge (search edge 3) 4 [3889] 3889 innerTail536_4
    (if 3 ≤ innerCandidates536_3.length then search edge 3 [3889, 6007] innerCandidates536_3 9896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan536_4

def innerTail536_2 : List Nat := 5821 :: innerTail536_3
def innerCandidates536_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates536_2_checked : candidates edge 4 3889 5821 innerTail536_3 best.bound = innerCandidates536_2 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan536_2 : scan edge (search edge 3) 4 [3889] 3889 innerTail536_2 best = best := by
  rw [innerTail536_2, scan, if_neg (by decide), innerCandidates536_2_checked]
  change scan edge (search edge 3) 4 [3889] 3889 innerTail536_3
    (if 3 ≤ innerCandidates536_2.length then search edge 3 [3889, 5821] innerCandidates536_2 9710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan536_3

def innerTail536_1 : List Nat := 5749 :: innerTail536_2
def innerCandidates536_1 : List Nat := [6007]
set_option maxRecDepth 4096 in
theorem innerCandidates536_1_checked : candidates edge 4 3889 5749 innerTail536_2 best.bound = innerCandidates536_1 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan536_1 : scan edge (search edge 3) 4 [3889] 3889 innerTail536_1 best = best := by
  rw [innerTail536_1, scan, if_neg (by decide), innerCandidates536_1_checked]
  change scan edge (search edge 3) 4 [3889] 3889 innerTail536_2
    (if 3 ≤ innerCandidates536_1.length then search edge 3 [3889, 5749] innerCandidates536_1 9638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan536_2

def innerTail536_0 : List Nat := 4021 :: innerTail536_1
def innerCandidates536_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates536_0_checked : candidates edge 4 3889 4021 innerTail536_1 best.bound = innerCandidates536_0 := by
  unfold candidates
  have row : edge 4021 = SearchData.row4021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan536_0 : scan edge (search edge 3) 4 [3889] 3889 innerTail536_0 best = best := by
  rw [innerTail536_0, scan, if_neg (by decide), innerCandidates536_0_checked]
  change scan edge (search edge 3) 4 [3889] 3889 innerTail536_1
    (if 3 ≤ innerCandidates536_0.length then search edge 3 [3889, 4021] innerCandidates536_0 7910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan536_1

theorem rootBranch536 : search edge 4 [3889] rootCandidates536 3889 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3889] 3889 innerTail536_0 best = best
  exact innerScan536_0

end ProjectEulerSolutions.PrimePairUpper.Search
