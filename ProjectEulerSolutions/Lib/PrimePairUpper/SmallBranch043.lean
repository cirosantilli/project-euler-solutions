import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates704 : List Nat := [5647, 7027, 7789, 8221, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates704_checked : candidates edge 5 0 5347 tail705 best.bound = rootCandidates704 := by
  unfold candidates
  have row : edge 5347 = SearchData.row5347.contains := by funext q; rfl
  rw [row]
  decide

def innerTail704_5 : List Nat := []
theorem innerScan704_5 : scan edge (search edge 3) 4 [5347] 5347 innerTail704_5 best = best := rfl

def innerTail704_4 : List Nat := 8287 :: innerTail704_5
def innerCandidates704_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates704_4_checked : candidates edge 4 5347 8287 innerTail704_5 best.bound = innerCandidates704_4 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan704_4 : scan edge (search edge 3) 4 [5347] 5347 innerTail704_4 best = best := by
  rw [innerTail704_4, scan, if_neg (by decide), innerCandidates704_4_checked]
  change scan edge (search edge 3) 4 [5347] 5347 innerTail704_5
    (if 3 ≤ innerCandidates704_4.length then search edge 3 [5347, 8287] innerCandidates704_4 13634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan704_5

def innerTail704_3 : List Nat := 8221 :: innerTail704_4
def innerCandidates704_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates704_3_checked : candidates edge 4 5347 8221 innerTail704_4 best.bound = innerCandidates704_3 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan704_3 : scan edge (search edge 3) 4 [5347] 5347 innerTail704_3 best = best := by
  rw [innerTail704_3, scan, if_neg (by decide), innerCandidates704_3_checked]
  change scan edge (search edge 3) 4 [5347] 5347 innerTail704_4
    (if 3 ≤ innerCandidates704_3.length then search edge 3 [5347, 8221] innerCandidates704_3 13568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan704_4

def innerTail704_2 : List Nat := 7789 :: innerTail704_3
def innerCandidates704_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates704_2_checked : candidates edge 4 5347 7789 innerTail704_3 best.bound = innerCandidates704_2 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan704_2 : scan edge (search edge 3) 4 [5347] 5347 innerTail704_2 best = best := by
  rw [innerTail704_2, scan, if_neg (by decide), innerCandidates704_2_checked]
  change scan edge (search edge 3) 4 [5347] 5347 innerTail704_3
    (if 3 ≤ innerCandidates704_2.length then search edge 3 [5347, 7789] innerCandidates704_2 13136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan704_3

def innerTail704_1 : List Nat := 7027 :: innerTail704_2
def innerCandidates704_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates704_1_checked : candidates edge 4 5347 7027 innerTail704_2 best.bound = innerCandidates704_1 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan704_1 : scan edge (search edge 3) 4 [5347] 5347 innerTail704_1 best = best := by
  rw [innerTail704_1, scan, if_neg (by decide), innerCandidates704_1_checked]
  change scan edge (search edge 3) 4 [5347] 5347 innerTail704_2
    (if 3 ≤ innerCandidates704_1.length then search edge 3 [5347, 7027] innerCandidates704_1 12374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan704_2

def innerTail704_0 : List Nat := 5647 :: innerTail704_1
def innerCandidates704_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates704_0_checked : candidates edge 4 5347 5647 innerTail704_1 best.bound = innerCandidates704_0 := by
  unfold candidates
  have row : edge 5647 = SearchData.row5647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan704_0 : scan edge (search edge 3) 4 [5347] 5347 innerTail704_0 best = best := by
  rw [innerTail704_0, scan, if_neg (by decide), innerCandidates704_0_checked]
  change scan edge (search edge 3) 4 [5347] 5347 innerTail704_1
    (if 3 ≤ innerCandidates704_0.length then search edge 3 [5347, 5647] innerCandidates704_0 10994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan704_1

theorem rootBranch704 : search edge 4 [5347] rootCandidates704 5347 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5347] 5347 innerTail704_0 best = best
  exact innerScan704_0

def rootCandidates703 : List Nat := [5657, 6719, 7283, 7649, 7703, 7901, 8009, 8093]
set_option maxRecDepth 4096 in
theorem rootCandidates703_checked : candidates edge 5 0 5333 tail704 best.bound = rootCandidates703 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext q; rfl
  rw [row]
  decide

def innerTail703_8 : List Nat := []
theorem innerScan703_8 : scan edge (search edge 3) 4 [5333] 5333 innerTail703_8 best = best := rfl

def innerTail703_7 : List Nat := 8093 :: innerTail703_8
def innerCandidates703_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates703_7_checked : candidates edge 4 5333 8093 innerTail703_8 best.bound = innerCandidates703_7 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan703_7 : scan edge (search edge 3) 4 [5333] 5333 innerTail703_7 best = best := by
  rw [innerTail703_7, scan, if_neg (by decide), innerCandidates703_7_checked]
  change scan edge (search edge 3) 4 [5333] 5333 innerTail703_8
    (if 3 ≤ innerCandidates703_7.length then search edge 3 [5333, 8093] innerCandidates703_7 13426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan703_8

def innerTail703_6 : List Nat := 8009 :: innerTail703_7
def innerCandidates703_6 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates703_6_checked : candidates edge 4 5333 8009 innerTail703_7 best.bound = innerCandidates703_6 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan703_6 : scan edge (search edge 3) 4 [5333] 5333 innerTail703_6 best = best := by
  rw [innerTail703_6, scan, if_neg (by decide), innerCandidates703_6_checked]
  change scan edge (search edge 3) 4 [5333] 5333 innerTail703_7
    (if 3 ≤ innerCandidates703_6.length then search edge 3 [5333, 8009] innerCandidates703_6 13342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan703_7

def innerTail703_5 : List Nat := 7901 :: innerTail703_6
def innerCandidates703_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates703_5_checked : candidates edge 4 5333 7901 innerTail703_6 best.bound = innerCandidates703_5 := by
  unfold candidates
  have row : edge 7901 = SearchData.row7901.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan703_5 : scan edge (search edge 3) 4 [5333] 5333 innerTail703_5 best = best := by
  rw [innerTail703_5, scan, if_neg (by decide), innerCandidates703_5_checked]
  change scan edge (search edge 3) 4 [5333] 5333 innerTail703_6
    (if 3 ≤ innerCandidates703_5.length then search edge 3 [5333, 7901] innerCandidates703_5 13234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan703_6

def innerTail703_4 : List Nat := 7703 :: innerTail703_5
def innerCandidates703_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates703_4_checked : candidates edge 4 5333 7703 innerTail703_5 best.bound = innerCandidates703_4 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan703_4 : scan edge (search edge 3) 4 [5333] 5333 innerTail703_4 best = best := by
  rw [innerTail703_4, scan, if_neg (by decide), innerCandidates703_4_checked]
  change scan edge (search edge 3) 4 [5333] 5333 innerTail703_5
    (if 3 ≤ innerCandidates703_4.length then search edge 3 [5333, 7703] innerCandidates703_4 13036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan703_5

def innerTail703_3 : List Nat := 7649 :: innerTail703_4
def innerCandidates703_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates703_3_checked : candidates edge 4 5333 7649 innerTail703_4 best.bound = innerCandidates703_3 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan703_3 : scan edge (search edge 3) 4 [5333] 5333 innerTail703_3 best = best := by
  rw [innerTail703_3, scan, if_neg (by decide), innerCandidates703_3_checked]
  change scan edge (search edge 3) 4 [5333] 5333 innerTail703_4
    (if 3 ≤ innerCandidates703_3.length then search edge 3 [5333, 7649] innerCandidates703_3 12982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan703_4

def innerTail703_2 : List Nat := 7283 :: innerTail703_3
def innerCandidates703_2 : List Nat := [7649, 7901]
set_option maxRecDepth 4096 in
theorem innerCandidates703_2_checked : candidates edge 4 5333 7283 innerTail703_3 best.bound = innerCandidates703_2 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan703_2 : scan edge (search edge 3) 4 [5333] 5333 innerTail703_2 best = best := by
  rw [innerTail703_2, scan, if_neg (by decide), innerCandidates703_2_checked]
  change scan edge (search edge 3) 4 [5333] 5333 innerTail703_3
    (if 3 ≤ innerCandidates703_2.length then search edge 3 [5333, 7283] innerCandidates703_2 12616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan703_3

def innerTail703_1 : List Nat := 6719 :: innerTail703_2
def innerCandidates703_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates703_1_checked : candidates edge 4 5333 6719 innerTail703_2 best.bound = innerCandidates703_1 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan703_1 : scan edge (search edge 3) 4 [5333] 5333 innerTail703_1 best = best := by
  rw [innerTail703_1, scan, if_neg (by decide), innerCandidates703_1_checked]
  change scan edge (search edge 3) 4 [5333] 5333 innerTail703_2
    (if 3 ≤ innerCandidates703_1.length then search edge 3 [5333, 6719] innerCandidates703_1 12052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan703_2

def innerTail703_0 : List Nat := 5657 :: innerTail703_1
def innerCandidates703_0 : List Nat := [8009]
set_option maxRecDepth 4096 in
theorem innerCandidates703_0_checked : candidates edge 4 5333 5657 innerTail703_1 best.bound = innerCandidates703_0 := by
  unfold candidates
  have row : edge 5657 = SearchData.row5657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan703_0 : scan edge (search edge 3) 4 [5333] 5333 innerTail703_0 best = best := by
  rw [innerTail703_0, scan, if_neg (by decide), innerCandidates703_0_checked]
  change scan edge (search edge 3) 4 [5333] 5333 innerTail703_1
    (if 3 ≤ innerCandidates703_0.length then search edge 3 [5333, 5657] innerCandidates703_0 10990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan703_1

theorem rootBranch703 : search edge 4 [5333] rootCandidates703 5333 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5333] 5333 innerTail703_0 best = best
  exact innerScan703_0

def rootCandidates702 : List Nat := [5521, 5527, 5659, 5779, 6037, 6067, 6361, 6397, 6481, 7237, 7333, 7537, 8089]
set_option maxRecDepth 4096 in
theorem rootCandidates702_checked : candidates edge 5 0 5323 tail703 best.bound = rootCandidates702 := by
  unfold candidates
  have row : edge 5323 = SearchData.row5323.contains := by funext q; rfl
  rw [row]
  decide

def innerTail702_13 : List Nat := []
theorem innerScan702_13 : scan edge (search edge 3) 4 [5323] 5323 innerTail702_13 best = best := rfl

def innerTail702_12 : List Nat := 8089 :: innerTail702_13
def innerCandidates702_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates702_12_checked : candidates edge 4 5323 8089 innerTail702_13 best.bound = innerCandidates702_12 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan702_12 : scan edge (search edge 3) 4 [5323] 5323 innerTail702_12 best = best := by
  rw [innerTail702_12, scan, if_neg (by decide), innerCandidates702_12_checked]
  change scan edge (search edge 3) 4 [5323] 5323 innerTail702_13
    (if 3 ≤ innerCandidates702_12.length then search edge 3 [5323, 8089] innerCandidates702_12 13412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan702_13

def innerTail702_11 : List Nat := 7537 :: innerTail702_12
def innerCandidates702_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates702_11_checked : candidates edge 4 5323 7537 innerTail702_12 best.bound = innerCandidates702_11 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan702_11 : scan edge (search edge 3) 4 [5323] 5323 innerTail702_11 best = best := by
  rw [innerTail702_11, scan, if_neg (by decide), innerCandidates702_11_checked]
  change scan edge (search edge 3) 4 [5323] 5323 innerTail702_12
    (if 3 ≤ innerCandidates702_11.length then search edge 3 [5323, 7537] innerCandidates702_11 12860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan702_12

def innerTail702_10 : List Nat := 7333 :: innerTail702_11
def innerCandidates702_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates702_10_checked : candidates edge 4 5323 7333 innerTail702_11 best.bound = innerCandidates702_10 := by
  unfold candidates
  have row : edge 7333 = SearchData.row7333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan702_10 : scan edge (search edge 3) 4 [5323] 5323 innerTail702_10 best = best := by
  rw [innerTail702_10, scan, if_neg (by decide), innerCandidates702_10_checked]
  change scan edge (search edge 3) 4 [5323] 5323 innerTail702_11
    (if 3 ≤ innerCandidates702_10.length then search edge 3 [5323, 7333] innerCandidates702_10 12656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan702_11

def innerTail702_9 : List Nat := 7237 :: innerTail702_10
def innerCandidates702_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates702_9_checked : candidates edge 4 5323 7237 innerTail702_10 best.bound = innerCandidates702_9 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan702_9 : scan edge (search edge 3) 4 [5323] 5323 innerTail702_9 best = best := by
  rw [innerTail702_9, scan, if_neg (by decide), innerCandidates702_9_checked]
  change scan edge (search edge 3) 4 [5323] 5323 innerTail702_10
    (if 3 ≤ innerCandidates702_9.length then search edge 3 [5323, 7237] innerCandidates702_9 12560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan702_10

def innerTail702_8 : List Nat := 6481 :: innerTail702_9
def innerCandidates702_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates702_8_checked : candidates edge 4 5323 6481 innerTail702_9 best.bound = innerCandidates702_8 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan702_8 : scan edge (search edge 3) 4 [5323] 5323 innerTail702_8 best = best := by
  rw [innerTail702_8, scan, if_neg (by decide), innerCandidates702_8_checked]
  change scan edge (search edge 3) 4 [5323] 5323 innerTail702_9
    (if 3 ≤ innerCandidates702_8.length then search edge 3 [5323, 6481] innerCandidates702_8 11804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan702_9

def innerTail702_7 : List Nat := 6397 :: innerTail702_8
def innerCandidates702_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates702_7_checked : candidates edge 4 5323 6397 innerTail702_8 best.bound = innerCandidates702_7 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan702_7 : scan edge (search edge 3) 4 [5323] 5323 innerTail702_7 best = best := by
  rw [innerTail702_7, scan, if_neg (by decide), innerCandidates702_7_checked]
  change scan edge (search edge 3) 4 [5323] 5323 innerTail702_8
    (if 3 ≤ innerCandidates702_7.length then search edge 3 [5323, 6397] innerCandidates702_7 11720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan702_8

def innerTail702_6 : List Nat := 6361 :: innerTail702_7
def innerCandidates702_6 : List Nat := [6481]
set_option maxRecDepth 4096 in
theorem innerCandidates702_6_checked : candidates edge 4 5323 6361 innerTail702_7 best.bound = innerCandidates702_6 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan702_6 : scan edge (search edge 3) 4 [5323] 5323 innerTail702_6 best = best := by
  rw [innerTail702_6, scan, if_neg (by decide), innerCandidates702_6_checked]
  change scan edge (search edge 3) 4 [5323] 5323 innerTail702_7
    (if 3 ≤ innerCandidates702_6.length then search edge 3 [5323, 6361] innerCandidates702_6 11684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan702_7

def innerTail702_5 : List Nat := 6067 :: innerTail702_6
def innerCandidates702_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates702_5_checked : candidates edge 4 5323 6067 innerTail702_6 best.bound = innerCandidates702_5 := by
  unfold candidates
  have row : edge 6067 = SearchData.row6067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan702_5 : scan edge (search edge 3) 4 [5323] 5323 innerTail702_5 best = best := by
  rw [innerTail702_5, scan, if_neg (by decide), innerCandidates702_5_checked]
  change scan edge (search edge 3) 4 [5323] 5323 innerTail702_6
    (if 3 ≤ innerCandidates702_5.length then search edge 3 [5323, 6067] innerCandidates702_5 11390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan702_6

def innerTail702_4 : List Nat := 6037 :: innerTail702_5
def innerCandidates702_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates702_4_checked : candidates edge 4 5323 6037 innerTail702_5 best.bound = innerCandidates702_4 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan702_4 : scan edge (search edge 3) 4 [5323] 5323 innerTail702_4 best = best := by
  rw [innerTail702_4, scan, if_neg (by decide), innerCandidates702_4_checked]
  change scan edge (search edge 3) 4 [5323] 5323 innerTail702_5
    (if 3 ≤ innerCandidates702_4.length then search edge 3 [5323, 6037] innerCandidates702_4 11360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan702_5

def innerTail702_3 : List Nat := 5779 :: innerTail702_4
def innerCandidates702_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates702_3_checked : candidates edge 4 5323 5779 innerTail702_4 best.bound = innerCandidates702_3 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan702_3 : scan edge (search edge 3) 4 [5323] 5323 innerTail702_3 best = best := by
  rw [innerTail702_3, scan, if_neg (by decide), innerCandidates702_3_checked]
  change scan edge (search edge 3) 4 [5323] 5323 innerTail702_4
    (if 3 ≤ innerCandidates702_3.length then search edge 3 [5323, 5779] innerCandidates702_3 11102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan702_4

def innerTail702_2 : List Nat := 5659 :: innerTail702_3
def innerCandidates702_2 : List Nat := [5779]
set_option maxRecDepth 4096 in
theorem innerCandidates702_2_checked : candidates edge 4 5323 5659 innerTail702_3 best.bound = innerCandidates702_2 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan702_2 : scan edge (search edge 3) 4 [5323] 5323 innerTail702_2 best = best := by
  rw [innerTail702_2, scan, if_neg (by decide), innerCandidates702_2_checked]
  change scan edge (search edge 3) 4 [5323] 5323 innerTail702_3
    (if 3 ≤ innerCandidates702_2.length then search edge 3 [5323, 5659] innerCandidates702_2 10982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan702_3

def innerTail702_1 : List Nat := 5527 :: innerTail702_2
def innerCandidates702_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates702_1_checked : candidates edge 4 5323 5527 innerTail702_2 best.bound = innerCandidates702_1 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan702_1 : scan edge (search edge 3) 4 [5323] 5323 innerTail702_1 best = best := by
  rw [innerTail702_1, scan, if_neg (by decide), innerCandidates702_1_checked]
  change scan edge (search edge 3) 4 [5323] 5323 innerTail702_2
    (if 3 ≤ innerCandidates702_1.length then search edge 3 [5323, 5527] innerCandidates702_1 10850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan702_2

def innerTail702_0 : List Nat := 5521 :: innerTail702_1
def innerCandidates702_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates702_0_checked : candidates edge 4 5323 5521 innerTail702_1 best.bound = innerCandidates702_0 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan702_0 : scan edge (search edge 3) 4 [5323] 5323 innerTail702_0 best = best := by
  rw [innerTail702_0, scan, if_neg (by decide), innerCandidates702_0_checked]
  change scan edge (search edge 3) 4 [5323] 5323 innerTail702_1
    (if 3 ≤ innerCandidates702_0.length then search edge 3 [5323, 5521] innerCandidates702_0 10844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan702_1

theorem rootBranch702 : search edge 4 [5323] rootCandidates702 5323 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5323] 5323 innerTail702_0 best = best
  exact innerScan702_0

def rootCandidates701 : List Nat := [5501, 6011, 6257, 6359, 6977, 7151, 7283, 7481, 8117, 8147]
set_option maxRecDepth 4096 in
theorem rootCandidates701_checked : candidates edge 5 0 5309 tail702 best.bound = rootCandidates701 := by
  unfold candidates
  have row : edge 5309 = SearchData.row5309.contains := by funext q; rfl
  rw [row]
  decide

def innerTail701_10 : List Nat := []
theorem innerScan701_10 : scan edge (search edge 3) 4 [5309] 5309 innerTail701_10 best = best := rfl

def innerTail701_9 : List Nat := 8147 :: innerTail701_10
def innerCandidates701_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates701_9_checked : candidates edge 4 5309 8147 innerTail701_10 best.bound = innerCandidates701_9 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan701_9 : scan edge (search edge 3) 4 [5309] 5309 innerTail701_9 best = best := by
  rw [innerTail701_9, scan, if_neg (by decide), innerCandidates701_9_checked]
  change scan edge (search edge 3) 4 [5309] 5309 innerTail701_10
    (if 3 ≤ innerCandidates701_9.length then search edge 3 [5309, 8147] innerCandidates701_9 13456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan701_10

def innerTail701_8 : List Nat := 8117 :: innerTail701_9
def innerCandidates701_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates701_8_checked : candidates edge 4 5309 8117 innerTail701_9 best.bound = innerCandidates701_8 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan701_8 : scan edge (search edge 3) 4 [5309] 5309 innerTail701_8 best = best := by
  rw [innerTail701_8, scan, if_neg (by decide), innerCandidates701_8_checked]
  change scan edge (search edge 3) 4 [5309] 5309 innerTail701_9
    (if 3 ≤ innerCandidates701_8.length then search edge 3 [5309, 8117] innerCandidates701_8 13426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan701_9

def innerTail701_7 : List Nat := 7481 :: innerTail701_8
def innerCandidates701_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates701_7_checked : candidates edge 4 5309 7481 innerTail701_8 best.bound = innerCandidates701_7 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan701_7 : scan edge (search edge 3) 4 [5309] 5309 innerTail701_7 best = best := by
  rw [innerTail701_7, scan, if_neg (by decide), innerCandidates701_7_checked]
  change scan edge (search edge 3) 4 [5309] 5309 innerTail701_8
    (if 3 ≤ innerCandidates701_7.length then search edge 3 [5309, 7481] innerCandidates701_7 12790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan701_8

def innerTail701_6 : List Nat := 7283 :: innerTail701_7
def innerCandidates701_6 : List Nat := [8147]
set_option maxRecDepth 4096 in
theorem innerCandidates701_6_checked : candidates edge 4 5309 7283 innerTail701_7 best.bound = innerCandidates701_6 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan701_6 : scan edge (search edge 3) 4 [5309] 5309 innerTail701_6 best = best := by
  rw [innerTail701_6, scan, if_neg (by decide), innerCandidates701_6_checked]
  change scan edge (search edge 3) 4 [5309] 5309 innerTail701_7
    (if 3 ≤ innerCandidates701_6.length then search edge 3 [5309, 7283] innerCandidates701_6 12592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan701_7

def innerTail701_5 : List Nat := 7151 :: innerTail701_6
def innerCandidates701_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates701_5_checked : candidates edge 4 5309 7151 innerTail701_6 best.bound = innerCandidates701_5 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan701_5 : scan edge (search edge 3) 4 [5309] 5309 innerTail701_5 best = best := by
  rw [innerTail701_5, scan, if_neg (by decide), innerCandidates701_5_checked]
  change scan edge (search edge 3) 4 [5309] 5309 innerTail701_6
    (if 3 ≤ innerCandidates701_5.length then search edge 3 [5309, 7151] innerCandidates701_5 12460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan701_6

def innerTail701_4 : List Nat := 6977 :: innerTail701_5
def innerCandidates701_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates701_4_checked : candidates edge 4 5309 6977 innerTail701_5 best.bound = innerCandidates701_4 := by
  unfold candidates
  have row : edge 6977 = SearchData.row6977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan701_4 : scan edge (search edge 3) 4 [5309] 5309 innerTail701_4 best = best := by
  rw [innerTail701_4, scan, if_neg (by decide), innerCandidates701_4_checked]
  change scan edge (search edge 3) 4 [5309] 5309 innerTail701_5
    (if 3 ≤ innerCandidates701_4.length then search edge 3 [5309, 6977] innerCandidates701_4 12286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan701_5

def innerTail701_3 : List Nat := 6359 :: innerTail701_4
def innerCandidates701_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates701_3_checked : candidates edge 4 5309 6359 innerTail701_4 best.bound = innerCandidates701_3 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan701_3 : scan edge (search edge 3) 4 [5309] 5309 innerTail701_3 best = best := by
  rw [innerTail701_3, scan, if_neg (by decide), innerCandidates701_3_checked]
  change scan edge (search edge 3) 4 [5309] 5309 innerTail701_4
    (if 3 ≤ innerCandidates701_3.length then search edge 3 [5309, 6359] innerCandidates701_3 11668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan701_4

def innerTail701_2 : List Nat := 6257 :: innerTail701_3
def innerCandidates701_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates701_2_checked : candidates edge 4 5309 6257 innerTail701_3 best.bound = innerCandidates701_2 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan701_2 : scan edge (search edge 3) 4 [5309] 5309 innerTail701_2 best = best := by
  rw [innerTail701_2, scan, if_neg (by decide), innerCandidates701_2_checked]
  change scan edge (search edge 3) 4 [5309] 5309 innerTail701_3
    (if 3 ≤ innerCandidates701_2.length then search edge 3 [5309, 6257] innerCandidates701_2 11566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan701_3

def innerTail701_1 : List Nat := 6011 :: innerTail701_2
def innerCandidates701_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates701_1_checked : candidates edge 4 5309 6011 innerTail701_2 best.bound = innerCandidates701_1 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan701_1 : scan edge (search edge 3) 4 [5309] 5309 innerTail701_1 best = best := by
  rw [innerTail701_1, scan, if_neg (by decide), innerCandidates701_1_checked]
  change scan edge (search edge 3) 4 [5309] 5309 innerTail701_2
    (if 3 ≤ innerCandidates701_1.length then search edge 3 [5309, 6011] innerCandidates701_1 11320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan701_2

def innerTail701_0 : List Nat := 5501 :: innerTail701_1
def innerCandidates701_0 : List Nat := [6359, 7481]
set_option maxRecDepth 4096 in
theorem innerCandidates701_0_checked : candidates edge 4 5309 5501 innerTail701_1 best.bound = innerCandidates701_0 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan701_0 : scan edge (search edge 3) 4 [5309] 5309 innerTail701_0 best = best := by
  rw [innerTail701_0, scan, if_neg (by decide), innerCandidates701_0_checked]
  change scan edge (search edge 3) 4 [5309] 5309 innerTail701_1
    (if 3 ≤ innerCandidates701_0.length then search edge 3 [5309, 5501] innerCandidates701_0 10810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan701_1

theorem rootBranch701 : search edge 4 [5309] rootCandidates701 5309 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5309] 5309 innerTail701_0 best = best
  exact innerScan701_0

def rootCandidates700 : List Nat := [5849, 6101, 7823, 8123, 8243, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates700_checked : candidates edge 5 0 5303 tail701 best.bound = rootCandidates700 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext q; rfl
  rw [row]
  decide

def innerTail700_6 : List Nat := []
theorem innerScan700_6 : scan edge (search edge 3) 4 [5303] 5303 innerTail700_6 best = best := rfl

def innerTail700_5 : List Nat := 8387 :: innerTail700_6
def innerCandidates700_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates700_5_checked : candidates edge 4 5303 8387 innerTail700_6 best.bound = innerCandidates700_5 := by
  unfold candidates
  have row : edge 8387 = SearchData.row8387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan700_5 : scan edge (search edge 3) 4 [5303] 5303 innerTail700_5 best = best := by
  rw [innerTail700_5, scan, if_neg (by decide), innerCandidates700_5_checked]
  change scan edge (search edge 3) 4 [5303] 5303 innerTail700_6
    (if 3 ≤ innerCandidates700_5.length then search edge 3 [5303, 8387] innerCandidates700_5 13690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan700_6

def innerTail700_4 : List Nat := 8243 :: innerTail700_5
def innerCandidates700_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates700_4_checked : candidates edge 4 5303 8243 innerTail700_5 best.bound = innerCandidates700_4 := by
  unfold candidates
  have row : edge 8243 = SearchData.row8243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan700_4 : scan edge (search edge 3) 4 [5303] 5303 innerTail700_4 best = best := by
  rw [innerTail700_4, scan, if_neg (by decide), innerCandidates700_4_checked]
  change scan edge (search edge 3) 4 [5303] 5303 innerTail700_5
    (if 3 ≤ innerCandidates700_4.length then search edge 3 [5303, 8243] innerCandidates700_4 13546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan700_5

def innerTail700_3 : List Nat := 8123 :: innerTail700_4
def innerCandidates700_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates700_3_checked : candidates edge 4 5303 8123 innerTail700_4 best.bound = innerCandidates700_3 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan700_3 : scan edge (search edge 3) 4 [5303] 5303 innerTail700_3 best = best := by
  rw [innerTail700_3, scan, if_neg (by decide), innerCandidates700_3_checked]
  change scan edge (search edge 3) 4 [5303] 5303 innerTail700_4
    (if 3 ≤ innerCandidates700_3.length then search edge 3 [5303, 8123] innerCandidates700_3 13426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan700_4

def innerTail700_2 : List Nat := 7823 :: innerTail700_3
def innerCandidates700_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates700_2_checked : candidates edge 4 5303 7823 innerTail700_3 best.bound = innerCandidates700_2 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan700_2 : scan edge (search edge 3) 4 [5303] 5303 innerTail700_2 best = best := by
  rw [innerTail700_2, scan, if_neg (by decide), innerCandidates700_2_checked]
  change scan edge (search edge 3) 4 [5303] 5303 innerTail700_3
    (if 3 ≤ innerCandidates700_2.length then search edge 3 [5303, 7823] innerCandidates700_2 13126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan700_3

def innerTail700_1 : List Nat := 6101 :: innerTail700_2
def innerCandidates700_1 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates700_1_checked : candidates edge 4 5303 6101 innerTail700_2 best.bound = innerCandidates700_1 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan700_1 : scan edge (search edge 3) 4 [5303] 5303 innerTail700_1 best = best := by
  rw [innerTail700_1, scan, if_neg (by decide), innerCandidates700_1_checked]
  change scan edge (search edge 3) 4 [5303] 5303 innerTail700_2
    (if 3 ≤ innerCandidates700_1.length then search edge 3 [5303, 6101] innerCandidates700_1 11404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan700_2

def innerTail700_0 : List Nat := 5849 :: innerTail700_1
def innerCandidates700_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates700_0_checked : candidates edge 4 5303 5849 innerTail700_1 best.bound = innerCandidates700_0 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan700_0 : scan edge (search edge 3) 4 [5303] 5303 innerTail700_0 best = best := by
  rw [innerTail700_0, scan, if_neg (by decide), innerCandidates700_0_checked]
  change scan edge (search edge 3) 4 [5303] 5303 innerTail700_1
    (if 3 ≤ innerCandidates700_0.length then search edge 3 [5303, 5849] innerCandidates700_0 11152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan700_1

theorem rootBranch700 : search edge 4 [5303] rootCandidates700 5303 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5303] 5303 innerTail700_0 best = best
  exact innerScan700_0

def rootCandidates699 : List Nat := [5843, 6257, 6263, 6701, 7541, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates699_checked : candidates edge 5 0 5297 tail700 best.bound = rootCandidates699 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext q; rfl
  rw [row]
  decide

def innerTail699_6 : List Nat := []
theorem innerScan699_6 : scan edge (search edge 3) 4 [5297] 5297 innerTail699_6 best = best := rfl

def innerTail699_5 : List Nat := 8297 :: innerTail699_6
def innerCandidates699_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates699_5_checked : candidates edge 4 5297 8297 innerTail699_6 best.bound = innerCandidates699_5 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan699_5 : scan edge (search edge 3) 4 [5297] 5297 innerTail699_5 best = best := by
  rw [innerTail699_5, scan, if_neg (by decide), innerCandidates699_5_checked]
  change scan edge (search edge 3) 4 [5297] 5297 innerTail699_6
    (if 3 ≤ innerCandidates699_5.length then search edge 3 [5297, 8297] innerCandidates699_5 13594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan699_6

def innerTail699_4 : List Nat := 7541 :: innerTail699_5
def innerCandidates699_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates699_4_checked : candidates edge 4 5297 7541 innerTail699_5 best.bound = innerCandidates699_4 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan699_4 : scan edge (search edge 3) 4 [5297] 5297 innerTail699_4 best = best := by
  rw [innerTail699_4, scan, if_neg (by decide), innerCandidates699_4_checked]
  change scan edge (search edge 3) 4 [5297] 5297 innerTail699_5
    (if 3 ≤ innerCandidates699_4.length then search edge 3 [5297, 7541] innerCandidates699_4 12838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan699_5

def innerTail699_3 : List Nat := 6701 :: innerTail699_4
def innerCandidates699_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates699_3_checked : candidates edge 4 5297 6701 innerTail699_4 best.bound = innerCandidates699_3 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan699_3 : scan edge (search edge 3) 4 [5297] 5297 innerTail699_3 best = best := by
  rw [innerTail699_3, scan, if_neg (by decide), innerCandidates699_3_checked]
  change scan edge (search edge 3) 4 [5297] 5297 innerTail699_4
    (if 3 ≤ innerCandidates699_3.length then search edge 3 [5297, 6701] innerCandidates699_3 11998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan699_4

def innerTail699_2 : List Nat := 6263 :: innerTail699_3
def innerCandidates699_2 : List Nat := [8297]
set_option maxRecDepth 4096 in
theorem innerCandidates699_2_checked : candidates edge 4 5297 6263 innerTail699_3 best.bound = innerCandidates699_2 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan699_2 : scan edge (search edge 3) 4 [5297] 5297 innerTail699_2 best = best := by
  rw [innerTail699_2, scan, if_neg (by decide), innerCandidates699_2_checked]
  change scan edge (search edge 3) 4 [5297] 5297 innerTail699_3
    (if 3 ≤ innerCandidates699_2.length then search edge 3 [5297, 6263] innerCandidates699_2 11560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan699_3

def innerTail699_1 : List Nat := 6257 :: innerTail699_2
def innerCandidates699_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates699_1_checked : candidates edge 4 5297 6257 innerTail699_2 best.bound = innerCandidates699_1 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan699_1 : scan edge (search edge 3) 4 [5297] 5297 innerTail699_1 best = best := by
  rw [innerTail699_1, scan, if_neg (by decide), innerCandidates699_1_checked]
  change scan edge (search edge 3) 4 [5297] 5297 innerTail699_2
    (if 3 ≤ innerCandidates699_1.length then search edge 3 [5297, 6257] innerCandidates699_1 11554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan699_2

def innerTail699_0 : List Nat := 5843 :: innerTail699_1
def innerCandidates699_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates699_0_checked : candidates edge 4 5297 5843 innerTail699_1 best.bound = innerCandidates699_0 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan699_0 : scan edge (search edge 3) 4 [5297] 5297 innerTail699_0 best = best := by
  rw [innerTail699_0, scan, if_neg (by decide), innerCandidates699_0_checked]
  change scan edge (search edge 3) 4 [5297] 5297 innerTail699_1
    (if 3 ≤ innerCandidates699_0.length then search edge 3 [5297, 5843] innerCandidates699_0 11140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan699_1

theorem rootBranch699 : search edge 4 [5297] rootCandidates699 5297 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5297] 5297 innerTail699_0 best = best
  exact innerScan699_0

def rootCandidates698 : List Nat := [5437, 5869, 6121, 6619, 7159, 7489, 7873, 7951, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates698_checked : candidates edge 5 0 5281 tail699 best.bound = rootCandidates698 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext q; rfl
  rw [row]
  decide

def innerTail698_9 : List Nat := []
theorem innerScan698_9 : scan edge (search edge 3) 4 [5281] 5281 innerTail698_9 best = best := rfl

def innerTail698_8 : List Nat := 8287 :: innerTail698_9
def innerCandidates698_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates698_8_checked : candidates edge 4 5281 8287 innerTail698_9 best.bound = innerCandidates698_8 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan698_8 : scan edge (search edge 3) 4 [5281] 5281 innerTail698_8 best = best := by
  rw [innerTail698_8, scan, if_neg (by decide), innerCandidates698_8_checked]
  change scan edge (search edge 3) 4 [5281] 5281 innerTail698_9
    (if 3 ≤ innerCandidates698_8.length then search edge 3 [5281, 8287] innerCandidates698_8 13568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan698_9

def innerTail698_7 : List Nat := 7951 :: innerTail698_8
def innerCandidates698_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates698_7_checked : candidates edge 4 5281 7951 innerTail698_8 best.bound = innerCandidates698_7 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan698_7 : scan edge (search edge 3) 4 [5281] 5281 innerTail698_7 best = best := by
  rw [innerTail698_7, scan, if_neg (by decide), innerCandidates698_7_checked]
  change scan edge (search edge 3) 4 [5281] 5281 innerTail698_8
    (if 3 ≤ innerCandidates698_7.length then search edge 3 [5281, 7951] innerCandidates698_7 13232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan698_8

def innerTail698_6 : List Nat := 7873 :: innerTail698_7
def innerCandidates698_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates698_6_checked : candidates edge 4 5281 7873 innerTail698_7 best.bound = innerCandidates698_6 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan698_6 : scan edge (search edge 3) 4 [5281] 5281 innerTail698_6 best = best := by
  rw [innerTail698_6, scan, if_neg (by decide), innerCandidates698_6_checked]
  change scan edge (search edge 3) 4 [5281] 5281 innerTail698_7
    (if 3 ≤ innerCandidates698_6.length then search edge 3 [5281, 7873] innerCandidates698_6 13154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan698_7

def innerTail698_5 : List Nat := 7489 :: innerTail698_6
def innerCandidates698_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates698_5_checked : candidates edge 4 5281 7489 innerTail698_6 best.bound = innerCandidates698_5 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan698_5 : scan edge (search edge 3) 4 [5281] 5281 innerTail698_5 best = best := by
  rw [innerTail698_5, scan, if_neg (by decide), innerCandidates698_5_checked]
  change scan edge (search edge 3) 4 [5281] 5281 innerTail698_6
    (if 3 ≤ innerCandidates698_5.length then search edge 3 [5281, 7489] innerCandidates698_5 12770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan698_6

def innerTail698_4 : List Nat := 7159 :: innerTail698_5
def innerCandidates698_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates698_4_checked : candidates edge 4 5281 7159 innerTail698_5 best.bound = innerCandidates698_4 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan698_4 : scan edge (search edge 3) 4 [5281] 5281 innerTail698_4 best = best := by
  rw [innerTail698_4, scan, if_neg (by decide), innerCandidates698_4_checked]
  change scan edge (search edge 3) 4 [5281] 5281 innerTail698_5
    (if 3 ≤ innerCandidates698_4.length then search edge 3 [5281, 7159] innerCandidates698_4 12440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan698_5

def innerTail698_3 : List Nat := 6619 :: innerTail698_4
def innerCandidates698_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates698_3_checked : candidates edge 4 5281 6619 innerTail698_4 best.bound = innerCandidates698_3 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan698_3 : scan edge (search edge 3) 4 [5281] 5281 innerTail698_3 best = best := by
  rw [innerTail698_3, scan, if_neg (by decide), innerCandidates698_3_checked]
  change scan edge (search edge 3) 4 [5281] 5281 innerTail698_4
    (if 3 ≤ innerCandidates698_3.length then search edge 3 [5281, 6619] innerCandidates698_3 11900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan698_4

def innerTail698_2 : List Nat := 6121 :: innerTail698_3
def innerCandidates698_2 : List Nat := [6619, 7489]
set_option maxRecDepth 4096 in
theorem innerCandidates698_2_checked : candidates edge 4 5281 6121 innerTail698_3 best.bound = innerCandidates698_2 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan698_2 : scan edge (search edge 3) 4 [5281] 5281 innerTail698_2 best = best := by
  rw [innerTail698_2, scan, if_neg (by decide), innerCandidates698_2_checked]
  change scan edge (search edge 3) 4 [5281] 5281 innerTail698_3
    (if 3 ≤ innerCandidates698_2.length then search edge 3 [5281, 6121] innerCandidates698_2 11402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan698_3

def innerTail698_1 : List Nat := 5869 :: innerTail698_2
def innerCandidates698_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates698_1_checked : candidates edge 4 5281 5869 innerTail698_2 best.bound = innerCandidates698_1 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan698_1 : scan edge (search edge 3) 4 [5281] 5281 innerTail698_1 best = best := by
  rw [innerTail698_1, scan, if_neg (by decide), innerCandidates698_1_checked]
  change scan edge (search edge 3) 4 [5281] 5281 innerTail698_2
    (if 3 ≤ innerCandidates698_1.length then search edge 3 [5281, 5869] innerCandidates698_1 11150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan698_2

def innerTail698_0 : List Nat := 5437 :: innerTail698_1
def innerCandidates698_0 : List Nat := [7159]
set_option maxRecDepth 4096 in
theorem innerCandidates698_0_checked : candidates edge 4 5281 5437 innerTail698_1 best.bound = innerCandidates698_0 := by
  unfold candidates
  have row : edge 5437 = SearchData.row5437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan698_0 : scan edge (search edge 3) 4 [5281] 5281 innerTail698_0 best = best := by
  rw [innerTail698_0, scan, if_neg (by decide), innerCandidates698_0_checked]
  change scan edge (search edge 3) 4 [5281] 5281 innerTail698_1
    (if 3 ≤ innerCandidates698_0.length then search edge 3 [5281, 5437] innerCandidates698_0 10718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan698_1

theorem rootBranch698 : search edge 4 [5281] rootCandidates698 5281 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5281] 5281 innerTail698_0 best = best
  exact innerScan698_0

end ProjectEulerSolutions.PrimePairUpper.Search
