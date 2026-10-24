import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates438 : List Nat := [3329, 3449, 3797, 4373, 4703, 4871, 4931, 5333, 6257, 6737, 7187, 7307, 7907]
set_option maxRecDepth 4096 in
theorem rootCandidates438_checked : candidates edge 5 0 3083 tail439 best.bound = rootCandidates438 := by
  unfold candidates
  have row : edge 3083 = SearchData.row3083.contains := by funext q; rfl
  rw [row]
  decide

def innerTail438_13 : List Nat := []
theorem innerScan438_13 : scan edge (search edge 3) 4 [3083] 3083 innerTail438_13 best = best := rfl

def innerTail438_12 : List Nat := 7907 :: innerTail438_13
def innerCandidates438_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates438_12_checked : candidates edge 4 3083 7907 innerTail438_13 best.bound = innerCandidates438_12 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan438_12 : scan edge (search edge 3) 4 [3083] 3083 innerTail438_12 best = best := by
  rw [innerTail438_12, scan, if_neg (by decide), innerCandidates438_12_checked]
  change scan edge (search edge 3) 4 [3083] 3083 innerTail438_13
    (if 3 ≤ innerCandidates438_12.length then search edge 3 [3083, 7907] innerCandidates438_12 10990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan438_13

def innerTail438_11 : List Nat := 7307 :: innerTail438_12
def innerCandidates438_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates438_11_checked : candidates edge 4 3083 7307 innerTail438_12 best.bound = innerCandidates438_11 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan438_11 : scan edge (search edge 3) 4 [3083] 3083 innerTail438_11 best = best := by
  rw [innerTail438_11, scan, if_neg (by decide), innerCandidates438_11_checked]
  change scan edge (search edge 3) 4 [3083] 3083 innerTail438_12
    (if 3 ≤ innerCandidates438_11.length then search edge 3 [3083, 7307] innerCandidates438_11 10390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan438_12

def innerTail438_10 : List Nat := 7187 :: innerTail438_11
def innerCandidates438_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates438_10_checked : candidates edge 4 3083 7187 innerTail438_11 best.bound = innerCandidates438_10 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan438_10 : scan edge (search edge 3) 4 [3083] 3083 innerTail438_10 best = best := by
  rw [innerTail438_10, scan, if_neg (by decide), innerCandidates438_10_checked]
  change scan edge (search edge 3) 4 [3083] 3083 innerTail438_11
    (if 3 ≤ innerCandidates438_10.length then search edge 3 [3083, 7187] innerCandidates438_10 10270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan438_11

def innerTail438_9 : List Nat := 6737 :: innerTail438_10
def innerCandidates438_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates438_9_checked : candidates edge 4 3083 6737 innerTail438_10 best.bound = innerCandidates438_9 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan438_9 : scan edge (search edge 3) 4 [3083] 3083 innerTail438_9 best = best := by
  rw [innerTail438_9, scan, if_neg (by decide), innerCandidates438_9_checked]
  change scan edge (search edge 3) 4 [3083] 3083 innerTail438_10
    (if 3 ≤ innerCandidates438_9.length then search edge 3 [3083, 6737] innerCandidates438_9 9820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan438_10

def innerTail438_8 : List Nat := 6257 :: innerTail438_9
def innerCandidates438_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates438_8_checked : candidates edge 4 3083 6257 innerTail438_9 best.bound = innerCandidates438_8 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan438_8 : scan edge (search edge 3) 4 [3083] 3083 innerTail438_8 best = best := by
  rw [innerTail438_8, scan, if_neg (by decide), innerCandidates438_8_checked]
  change scan edge (search edge 3) 4 [3083] 3083 innerTail438_9
    (if 3 ≤ innerCandidates438_8.length then search edge 3 [3083, 6257] innerCandidates438_8 9340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan438_9

def innerTail438_7 : List Nat := 5333 :: innerTail438_8
def innerCandidates438_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates438_7_checked : candidates edge 4 3083 5333 innerTail438_8 best.bound = innerCandidates438_7 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan438_7 : scan edge (search edge 3) 4 [3083] 3083 innerTail438_7 best = best := by
  rw [innerTail438_7, scan, if_neg (by decide), innerCandidates438_7_checked]
  change scan edge (search edge 3) 4 [3083] 3083 innerTail438_8
    (if 3 ≤ innerCandidates438_7.length then search edge 3 [3083, 5333] innerCandidates438_7 8416 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan438_8

def innerTail438_6 : List Nat := 4931 :: innerTail438_7
def innerCandidates438_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates438_6_checked : candidates edge 4 3083 4931 innerTail438_7 best.bound = innerCandidates438_6 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan438_6 : scan edge (search edge 3) 4 [3083] 3083 innerTail438_6 best = best := by
  rw [innerTail438_6, scan, if_neg (by decide), innerCandidates438_6_checked]
  change scan edge (search edge 3) 4 [3083] 3083 innerTail438_7
    (if 3 ≤ innerCandidates438_6.length then search edge 3 [3083, 4931] innerCandidates438_6 8014 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan438_7

def innerTail438_5 : List Nat := 4871 :: innerTail438_6
def innerCandidates438_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates438_5_checked : candidates edge 4 3083 4871 innerTail438_6 best.bound = innerCandidates438_5 := by
  unfold candidates
  have row : edge 4871 = SearchData.row4871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan438_5 : scan edge (search edge 3) 4 [3083] 3083 innerTail438_5 best = best := by
  rw [innerTail438_5, scan, if_neg (by decide), innerCandidates438_5_checked]
  change scan edge (search edge 3) 4 [3083] 3083 innerTail438_6
    (if 3 ≤ innerCandidates438_5.length then search edge 3 [3083, 4871] innerCandidates438_5 7954 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan438_6

def innerTail438_4 : List Nat := 4703 :: innerTail438_5
def innerCandidates438_4 : List Nat := [6257]
set_option maxRecDepth 4096 in
theorem innerCandidates438_4_checked : candidates edge 4 3083 4703 innerTail438_5 best.bound = innerCandidates438_4 := by
  unfold candidates
  have row : edge 4703 = SearchData.row4703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan438_4 : scan edge (search edge 3) 4 [3083] 3083 innerTail438_4 best = best := by
  rw [innerTail438_4, scan, if_neg (by decide), innerCandidates438_4_checked]
  change scan edge (search edge 3) 4 [3083] 3083 innerTail438_5
    (if 3 ≤ innerCandidates438_4.length then search edge 3 [3083, 4703] innerCandidates438_4 7786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan438_5

def innerTail438_3 : List Nat := 4373 :: innerTail438_4
def innerCandidates438_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates438_3_checked : candidates edge 4 3083 4373 innerTail438_4 best.bound = innerCandidates438_3 := by
  unfold candidates
  have row : edge 4373 = SearchData.row4373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan438_3 : scan edge (search edge 3) 4 [3083] 3083 innerTail438_3 best = best := by
  rw [innerTail438_3, scan, if_neg (by decide), innerCandidates438_3_checked]
  change scan edge (search edge 3) 4 [3083] 3083 innerTail438_4
    (if 3 ≤ innerCandidates438_3.length then search edge 3 [3083, 4373] innerCandidates438_3 7456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan438_4

def innerTail438_2 : List Nat := 3797 :: innerTail438_3
def innerCandidates438_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates438_2_checked : candidates edge 4 3083 3797 innerTail438_3 best.bound = innerCandidates438_2 := by
  unfold candidates
  have row : edge 3797 = SearchData.row3797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan438_2 : scan edge (search edge 3) 4 [3083] 3083 innerTail438_2 best = best := by
  rw [innerTail438_2, scan, if_neg (by decide), innerCandidates438_2_checked]
  change scan edge (search edge 3) 4 [3083] 3083 innerTail438_3
    (if 3 ≤ innerCandidates438_2.length then search edge 3 [3083, 3797] innerCandidates438_2 6880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan438_3

def innerTail438_1 : List Nat := 3449 :: innerTail438_2
def innerCandidates438_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates438_1_checked : candidates edge 4 3083 3449 innerTail438_2 best.bound = innerCandidates438_1 := by
  unfold candidates
  have row : edge 3449 = SearchData.row3449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan438_1 : scan edge (search edge 3) 4 [3083] 3083 innerTail438_1 best = best := by
  rw [innerTail438_1, scan, if_neg (by decide), innerCandidates438_1_checked]
  change scan edge (search edge 3) 4 [3083] 3083 innerTail438_2
    (if 3 ≤ innerCandidates438_1.length then search edge 3 [3083, 3449] innerCandidates438_1 6532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan438_2

def innerTail438_0 : List Nat := 3329 :: innerTail438_1
def innerCandidates438_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates438_0_checked : candidates edge 4 3083 3329 innerTail438_1 best.bound = innerCandidates438_0 := by
  unfold candidates
  have row : edge 3329 = SearchData.row3329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan438_0 : scan edge (search edge 3) 4 [3083] 3083 innerTail438_0 best = best := by
  rw [innerTail438_0, scan, if_neg (by decide), innerCandidates438_0_checked]
  change scan edge (search edge 3) 4 [3083] 3083 innerTail438_1
    (if 3 ≤ innerCandidates438_0.length then search edge 3 [3083, 3329] innerCandidates438_0 6412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan438_1

theorem rootBranch438 : search edge 4 [3083] rootCandidates438 3083 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3083] 3083 innerTail438_0 best = best
  exact innerScan438_0

def rootCandidates437 : List Nat := [3181, 3673, 3709, 3877, 4273, 4591, 4957, 5641, 6301, 6679, 6709, 7297, 7561, 7699, 7867, 8161, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates437_checked : candidates edge 5 0 3079 tail438 best.bound = rootCandidates437 := by
  unfold candidates
  have row : edge 3079 = SearchData.row3079.contains := by funext q; rfl
  rw [row]
  decide

def innerTail437_17 : List Nat := []
theorem innerScan437_17 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_17 best = best := rfl

def innerTail437_16 : List Nat := 8329 :: innerTail437_17
def innerCandidates437_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates437_16_checked : candidates edge 4 3079 8329 innerTail437_17 best.bound = innerCandidates437_16 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan437_16 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_16 best = best := by
  rw [innerTail437_16, scan, if_neg (by decide), innerCandidates437_16_checked]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_17
    (if 3 ≤ innerCandidates437_16.length then search edge 3 [3079, 8329] innerCandidates437_16 11408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan437_17

def innerTail437_15 : List Nat := 8161 :: innerTail437_16
def innerCandidates437_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates437_15_checked : candidates edge 4 3079 8161 innerTail437_16 best.bound = innerCandidates437_15 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan437_15 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_15 best = best := by
  rw [innerTail437_15, scan, if_neg (by decide), innerCandidates437_15_checked]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_16
    (if 3 ≤ innerCandidates437_15.length then search edge 3 [3079, 8161] innerCandidates437_15 11240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan437_16

def innerTail437_14 : List Nat := 7867 :: innerTail437_15
def innerCandidates437_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates437_14_checked : candidates edge 4 3079 7867 innerTail437_15 best.bound = innerCandidates437_14 := by
  unfold candidates
  have row : edge 7867 = SearchData.row7867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan437_14 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_14 best = best := by
  rw [innerTail437_14, scan, if_neg (by decide), innerCandidates437_14_checked]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_15
    (if 3 ≤ innerCandidates437_14.length then search edge 3 [3079, 7867] innerCandidates437_14 10946 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan437_15

def innerTail437_13 : List Nat := 7699 :: innerTail437_14
def innerCandidates437_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates437_13_checked : candidates edge 4 3079 7699 innerTail437_14 best.bound = innerCandidates437_13 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan437_13 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_13 best = best := by
  rw [innerTail437_13, scan, if_neg (by decide), innerCandidates437_13_checked]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_14
    (if 3 ≤ innerCandidates437_13.length then search edge 3 [3079, 7699] innerCandidates437_13 10778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan437_14

def innerTail437_12 : List Nat := 7561 :: innerTail437_13
def innerCandidates437_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates437_12_checked : candidates edge 4 3079 7561 innerTail437_13 best.bound = innerCandidates437_12 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan437_12 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_12 best = best := by
  rw [innerTail437_12, scan, if_neg (by decide), innerCandidates437_12_checked]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_13
    (if 3 ≤ innerCandidates437_12.length then search edge 3 [3079, 7561] innerCandidates437_12 10640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan437_13

def innerTail437_11 : List Nat := 7297 :: innerTail437_12
def innerCandidates437_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates437_11_checked : candidates edge 4 3079 7297 innerTail437_12 best.bound = innerCandidates437_11 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan437_11 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_11 best = best := by
  rw [innerTail437_11, scan, if_neg (by decide), innerCandidates437_11_checked]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_12
    (if 3 ≤ innerCandidates437_11.length then search edge 3 [3079, 7297] innerCandidates437_11 10376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan437_12

def innerTail437_10 : List Nat := 6709 :: innerTail437_11
def innerCandidates437_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates437_10_checked : candidates edge 4 3079 6709 innerTail437_11 best.bound = innerCandidates437_10 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan437_10 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_10 best = best := by
  rw [innerTail437_10, scan, if_neg (by decide), innerCandidates437_10_checked]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_11
    (if 3 ≤ innerCandidates437_10.length then search edge 3 [3079, 6709] innerCandidates437_10 9788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan437_11

def innerTail437_9 : List Nat := 6679 :: innerTail437_10
def innerCandidates437_9 : List Nat := [8161]
set_option maxRecDepth 4096 in
theorem innerCandidates437_9_checked : candidates edge 4 3079 6679 innerTail437_10 best.bound = innerCandidates437_9 := by
  unfold candidates
  have row : edge 6679 = SearchData.row6679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan437_9 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_9 best = best := by
  rw [innerTail437_9, scan, if_neg (by decide), innerCandidates437_9_checked]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_10
    (if 3 ≤ innerCandidates437_9.length then search edge 3 [3079, 6679] innerCandidates437_9 9758 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan437_10

def innerTail437_8 : List Nat := 6301 :: innerTail437_9
def innerCandidates437_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates437_8_checked : candidates edge 4 3079 6301 innerTail437_9 best.bound = innerCandidates437_8 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan437_8 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_8 best = best := by
  rw [innerTail437_8, scan, if_neg (by decide), innerCandidates437_8_checked]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_9
    (if 3 ≤ innerCandidates437_8.length then search edge 3 [3079, 6301] innerCandidates437_8 9380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan437_9

def innerTail437_7 : List Nat := 5641 :: innerTail437_8
def innerCandidates437_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates437_7_checked : candidates edge 4 3079 5641 innerTail437_8 best.bound = innerCandidates437_7 := by
  unfold candidates
  have row : edge 5641 = SearchData.row5641.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan437_7 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_7 best = best := by
  rw [innerTail437_7, scan, if_neg (by decide), innerCandidates437_7_checked]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_8
    (if 3 ≤ innerCandidates437_7.length then search edge 3 [3079, 5641] innerCandidates437_7 8720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan437_8

def innerTail437_6 : List Nat := 4957 :: innerTail437_7
def innerCandidates437_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates437_6_checked : candidates edge 4 3079 4957 innerTail437_7 best.bound = innerCandidates437_6 := by
  unfold candidates
  have row : edge 4957 = SearchData.row4957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan437_6 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_6 best = best := by
  rw [innerTail437_6, scan, if_neg (by decide), innerCandidates437_6_checked]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_7
    (if 3 ≤ innerCandidates437_6.length then search edge 3 [3079, 4957] innerCandidates437_6 8036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan437_7

def innerTail437_5 : List Nat := 4591 :: innerTail437_6
def innerCandidates437_5 : List Nat := [8329]
set_option maxRecDepth 4096 in
theorem innerCandidates437_5_checked : candidates edge 4 3079 4591 innerTail437_6 best.bound = innerCandidates437_5 := by
  unfold candidates
  have row : edge 4591 = SearchData.row4591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan437_5 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_5 best = best := by
  rw [innerTail437_5, scan, if_neg (by decide), innerCandidates437_5_checked]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_6
    (if 3 ≤ innerCandidates437_5.length then search edge 3 [3079, 4591] innerCandidates437_5 7670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan437_6

def innerTail437_4 : List Nat := 4273 :: innerTail437_5
def innerCandidates437_4 : List Nat := [7297]
set_option maxRecDepth 4096 in
theorem innerCandidates437_4_checked : candidates edge 4 3079 4273 innerTail437_5 best.bound = innerCandidates437_4 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan437_4 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_4 best = best := by
  rw [innerTail437_4, scan, if_neg (by decide), innerCandidates437_4_checked]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_5
    (if 3 ≤ innerCandidates437_4.length then search edge 3 [3079, 4273] innerCandidates437_4 7352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan437_5

def innerTail437_3 : List Nat := 3877 :: innerTail437_4
def innerCandidates437_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates437_3_checked : candidates edge 4 3079 3877 innerTail437_4 best.bound = innerCandidates437_3 := by
  unfold candidates
  have row : edge 3877 = SearchData.row3877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan437_3 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_3 best = best := by
  rw [innerTail437_3, scan, if_neg (by decide), innerCandidates437_3_checked]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_4
    (if 3 ≤ innerCandidates437_3.length then search edge 3 [3079, 3877] innerCandidates437_3 6956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan437_4

def innerTail437_2 : List Nat := 3709 :: innerTail437_3
def innerCandidates437_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates437_2_checked : candidates edge 4 3079 3709 innerTail437_3 best.bound = innerCandidates437_2 := by
  unfold candidates
  have row : edge 3709 = SearchData.row3709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan437_2 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_2 best = best := by
  rw [innerTail437_2, scan, if_neg (by decide), innerCandidates437_2_checked]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_3
    (if 3 ≤ innerCandidates437_2.length then search edge 3 [3079, 3709] innerCandidates437_2 6788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan437_3

def innerTail437_1 : List Nat := 3673 :: innerTail437_2
def innerCandidates437_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates437_1_checked : candidates edge 4 3079 3673 innerTail437_2 best.bound = innerCandidates437_1 := by
  unfold candidates
  have row : edge 3673 = SearchData.row3673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan437_1 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_1 best = best := by
  rw [innerTail437_1, scan, if_neg (by decide), innerCandidates437_1_checked]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_2
    (if 3 ≤ innerCandidates437_1.length then search edge 3 [3079, 3673] innerCandidates437_1 6752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan437_2

def innerTail437_0 : List Nat := 3181 :: innerTail437_1
def innerCandidates437_0 : List Nat := [4273, 6709]
set_option maxRecDepth 4096 in
theorem innerCandidates437_0_checked : candidates edge 4 3079 3181 innerTail437_1 best.bound = innerCandidates437_0 := by
  unfold candidates
  have row : edge 3181 = SearchData.row3181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan437_0 : scan edge (search edge 3) 4 [3079] 3079 innerTail437_0 best = best := by
  rw [innerTail437_0, scan, if_neg (by decide), innerCandidates437_0_checked]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_1
    (if 3 ≤ innerCandidates437_0.length then search edge 3 [3079, 3181] innerCandidates437_0 6260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan437_1

theorem rootBranch437 : search edge 4 [3079] rootCandidates437 3079 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3079] 3079 innerTail437_0 best = best
  exact innerScan437_0

def rootCandidates436 : List Nat := [3373, 4909, 5167, 5413, 5419, 5737, 6661, 7459, 7489, 7687, 7753, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates436_checked : candidates edge 5 0 3067 tail437 best.bound = rootCandidates436 := by
  unfold candidates
  have row : edge 3067 = SearchData.row3067.contains := by funext q; rfl
  rw [row]
  decide

def innerTail436_12 : List Nat := []
theorem innerScan436_12 : scan edge (search edge 3) 4 [3067] 3067 innerTail436_12 best = best := rfl

def innerTail436_11 : List Nat := 8269 :: innerTail436_12
def innerCandidates436_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates436_11_checked : candidates edge 4 3067 8269 innerTail436_12 best.bound = innerCandidates436_11 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan436_11 : scan edge (search edge 3) 4 [3067] 3067 innerTail436_11 best = best := by
  rw [innerTail436_11, scan, if_neg (by decide), innerCandidates436_11_checked]
  change scan edge (search edge 3) 4 [3067] 3067 innerTail436_12
    (if 3 ≤ innerCandidates436_11.length then search edge 3 [3067, 8269] innerCandidates436_11 11336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan436_12

def innerTail436_10 : List Nat := 7753 :: innerTail436_11
def innerCandidates436_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates436_10_checked : candidates edge 4 3067 7753 innerTail436_11 best.bound = innerCandidates436_10 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan436_10 : scan edge (search edge 3) 4 [3067] 3067 innerTail436_10 best = best := by
  rw [innerTail436_10, scan, if_neg (by decide), innerCandidates436_10_checked]
  change scan edge (search edge 3) 4 [3067] 3067 innerTail436_11
    (if 3 ≤ innerCandidates436_10.length then search edge 3 [3067, 7753] innerCandidates436_10 10820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan436_11

def innerTail436_9 : List Nat := 7687 :: innerTail436_10
def innerCandidates436_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates436_9_checked : candidates edge 4 3067 7687 innerTail436_10 best.bound = innerCandidates436_9 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan436_9 : scan edge (search edge 3) 4 [3067] 3067 innerTail436_9 best = best := by
  rw [innerTail436_9, scan, if_neg (by decide), innerCandidates436_9_checked]
  change scan edge (search edge 3) 4 [3067] 3067 innerTail436_10
    (if 3 ≤ innerCandidates436_9.length then search edge 3 [3067, 7687] innerCandidates436_9 10754 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan436_10

def innerTail436_8 : List Nat := 7489 :: innerTail436_9
def innerCandidates436_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates436_8_checked : candidates edge 4 3067 7489 innerTail436_9 best.bound = innerCandidates436_8 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan436_8 : scan edge (search edge 3) 4 [3067] 3067 innerTail436_8 best = best := by
  rw [innerTail436_8, scan, if_neg (by decide), innerCandidates436_8_checked]
  change scan edge (search edge 3) 4 [3067] 3067 innerTail436_9
    (if 3 ≤ innerCandidates436_8.length then search edge 3 [3067, 7489] innerCandidates436_8 10556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan436_9

def innerTail436_7 : List Nat := 7459 :: innerTail436_8
def innerCandidates436_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates436_7_checked : candidates edge 4 3067 7459 innerTail436_8 best.bound = innerCandidates436_7 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan436_7 : scan edge (search edge 3) 4 [3067] 3067 innerTail436_7 best = best := by
  rw [innerTail436_7, scan, if_neg (by decide), innerCandidates436_7_checked]
  change scan edge (search edge 3) 4 [3067] 3067 innerTail436_8
    (if 3 ≤ innerCandidates436_7.length then search edge 3 [3067, 7459] innerCandidates436_7 10526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan436_8

def innerTail436_6 : List Nat := 6661 :: innerTail436_7
def innerCandidates436_6 : List Nat := [7687]
set_option maxRecDepth 4096 in
theorem innerCandidates436_6_checked : candidates edge 4 3067 6661 innerTail436_7 best.bound = innerCandidates436_6 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan436_6 : scan edge (search edge 3) 4 [3067] 3067 innerTail436_6 best = best := by
  rw [innerTail436_6, scan, if_neg (by decide), innerCandidates436_6_checked]
  change scan edge (search edge 3) 4 [3067] 3067 innerTail436_7
    (if 3 ≤ innerCandidates436_6.length then search edge 3 [3067, 6661] innerCandidates436_6 9728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan436_7

def innerTail436_5 : List Nat := 5737 :: innerTail436_6
def innerCandidates436_5 : List Nat := [7687]
set_option maxRecDepth 4096 in
theorem innerCandidates436_5_checked : candidates edge 4 3067 5737 innerTail436_6 best.bound = innerCandidates436_5 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan436_5 : scan edge (search edge 3) 4 [3067] 3067 innerTail436_5 best = best := by
  rw [innerTail436_5, scan, if_neg (by decide), innerCandidates436_5_checked]
  change scan edge (search edge 3) 4 [3067] 3067 innerTail436_6
    (if 3 ≤ innerCandidates436_5.length then search edge 3 [3067, 5737] innerCandidates436_5 8804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan436_6

def innerTail436_4 : List Nat := 5419 :: innerTail436_5
def innerCandidates436_4 : List Nat := [7687]
set_option maxRecDepth 4096 in
theorem innerCandidates436_4_checked : candidates edge 4 3067 5419 innerTail436_5 best.bound = innerCandidates436_4 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan436_4 : scan edge (search edge 3) 4 [3067] 3067 innerTail436_4 best = best := by
  rw [innerTail436_4, scan, if_neg (by decide), innerCandidates436_4_checked]
  change scan edge (search edge 3) 4 [3067] 3067 innerTail436_5
    (if 3 ≤ innerCandidates436_4.length then search edge 3 [3067, 5419] innerCandidates436_4 8486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan436_5

def innerTail436_3 : List Nat := 5413 :: innerTail436_4
def innerCandidates436_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates436_3_checked : candidates edge 4 3067 5413 innerTail436_4 best.bound = innerCandidates436_3 := by
  unfold candidates
  have row : edge 5413 = SearchData.row5413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan436_3 : scan edge (search edge 3) 4 [3067] 3067 innerTail436_3 best = best := by
  rw [innerTail436_3, scan, if_neg (by decide), innerCandidates436_3_checked]
  change scan edge (search edge 3) 4 [3067] 3067 innerTail436_4
    (if 3 ≤ innerCandidates436_3.length then search edge 3 [3067, 5413] innerCandidates436_3 8480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan436_4

def innerTail436_2 : List Nat := 5167 :: innerTail436_3
def innerCandidates436_2 : List Nat := [5737, 6661]
set_option maxRecDepth 4096 in
theorem innerCandidates436_2_checked : candidates edge 4 3067 5167 innerTail436_3 best.bound = innerCandidates436_2 := by
  unfold candidates
  have row : edge 5167 = SearchData.row5167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan436_2 : scan edge (search edge 3) 4 [3067] 3067 innerTail436_2 best = best := by
  rw [innerTail436_2, scan, if_neg (by decide), innerCandidates436_2_checked]
  change scan edge (search edge 3) 4 [3067] 3067 innerTail436_3
    (if 3 ≤ innerCandidates436_2.length then search edge 3 [3067, 5167] innerCandidates436_2 8234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan436_3

def innerTail436_1 : List Nat := 4909 :: innerTail436_2
def innerCandidates436_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates436_1_checked : candidates edge 4 3067 4909 innerTail436_2 best.bound = innerCandidates436_1 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan436_1 : scan edge (search edge 3) 4 [3067] 3067 innerTail436_1 best = best := by
  rw [innerTail436_1, scan, if_neg (by decide), innerCandidates436_1_checked]
  change scan edge (search edge 3) 4 [3067] 3067 innerTail436_2
    (if 3 ≤ innerCandidates436_1.length then search edge 3 [3067, 4909] innerCandidates436_1 7976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan436_2

def innerTail436_0 : List Nat := 3373 :: innerTail436_1
def innerCandidates436_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates436_0_checked : candidates edge 4 3067 3373 innerTail436_1 best.bound = innerCandidates436_0 := by
  unfold candidates
  have row : edge 3373 = SearchData.row3373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan436_0 : scan edge (search edge 3) 4 [3067] 3067 innerTail436_0 best = best := by
  rw [innerTail436_0, scan, if_neg (by decide), innerCandidates436_0_checked]
  change scan edge (search edge 3) 4 [3067] 3067 innerTail436_1
    (if 3 ≤ innerCandidates436_0.length then search edge 3 [3067, 3373] innerCandidates436_0 6440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan436_1

theorem rootBranch436 : search edge 4 [3067] rootCandidates436 3067 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3067] 3067 innerTail436_0 best = best
  exact innerScan436_0

def rootCandidates435 : List Nat := [3571, 3607, 3613, 4729, 5437, 5857, 6343, 6673, 6829, 6907, 8167, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates435_checked : candidates edge 5 0 3061 tail436 best.bound = rootCandidates435 := by
  unfold candidates
  have row : edge 3061 = SearchData.row3061.contains := by funext q; rfl
  rw [row]
  decide

def innerTail435_12 : List Nat := []
theorem innerScan435_12 : scan edge (search edge 3) 4 [3061] 3061 innerTail435_12 best = best := rfl

def innerTail435_11 : List Nat := 8293 :: innerTail435_12
def innerCandidates435_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates435_11_checked : candidates edge 4 3061 8293 innerTail435_12 best.bound = innerCandidates435_11 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan435_11 : scan edge (search edge 3) 4 [3061] 3061 innerTail435_11 best = best := by
  rw [innerTail435_11, scan, if_neg (by decide), innerCandidates435_11_checked]
  change scan edge (search edge 3) 4 [3061] 3061 innerTail435_12
    (if 3 ≤ innerCandidates435_11.length then search edge 3 [3061, 8293] innerCandidates435_11 11354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan435_12

def innerTail435_10 : List Nat := 8167 :: innerTail435_11
def innerCandidates435_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates435_10_checked : candidates edge 4 3061 8167 innerTail435_11 best.bound = innerCandidates435_10 := by
  unfold candidates
  have row : edge 8167 = SearchData.row8167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan435_10 : scan edge (search edge 3) 4 [3061] 3061 innerTail435_10 best = best := by
  rw [innerTail435_10, scan, if_neg (by decide), innerCandidates435_10_checked]
  change scan edge (search edge 3) 4 [3061] 3061 innerTail435_11
    (if 3 ≤ innerCandidates435_10.length then search edge 3 [3061, 8167] innerCandidates435_10 11228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan435_11

def innerTail435_9 : List Nat := 6907 :: innerTail435_10
def innerCandidates435_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates435_9_checked : candidates edge 4 3061 6907 innerTail435_10 best.bound = innerCandidates435_9 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan435_9 : scan edge (search edge 3) 4 [3061] 3061 innerTail435_9 best = best := by
  rw [innerTail435_9, scan, if_neg (by decide), innerCandidates435_9_checked]
  change scan edge (search edge 3) 4 [3061] 3061 innerTail435_10
    (if 3 ≤ innerCandidates435_9.length then search edge 3 [3061, 6907] innerCandidates435_9 9968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan435_10

def innerTail435_8 : List Nat := 6829 :: innerTail435_9
def innerCandidates435_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates435_8_checked : candidates edge 4 3061 6829 innerTail435_9 best.bound = innerCandidates435_8 := by
  unfold candidates
  have row : edge 6829 = SearchData.row6829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan435_8 : scan edge (search edge 3) 4 [3061] 3061 innerTail435_8 best = best := by
  rw [innerTail435_8, scan, if_neg (by decide), innerCandidates435_8_checked]
  change scan edge (search edge 3) 4 [3061] 3061 innerTail435_9
    (if 3 ≤ innerCandidates435_8.length then search edge 3 [3061, 6829] innerCandidates435_8 9890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan435_9

def innerTail435_7 : List Nat := 6673 :: innerTail435_8
def innerCandidates435_7 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates435_7_checked : candidates edge 4 3061 6673 innerTail435_8 best.bound = innerCandidates435_7 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan435_7 : scan edge (search edge 3) 4 [3061] 3061 innerTail435_7 best = best := by
  rw [innerTail435_7, scan, if_neg (by decide), innerCandidates435_7_checked]
  change scan edge (search edge 3) 4 [3061] 3061 innerTail435_8
    (if 3 ≤ innerCandidates435_7.length then search edge 3 [3061, 6673] innerCandidates435_7 9734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan435_8

def innerTail435_6 : List Nat := 6343 :: innerTail435_7
def innerCandidates435_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates435_6_checked : candidates edge 4 3061 6343 innerTail435_7 best.bound = innerCandidates435_6 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan435_6 : scan edge (search edge 3) 4 [3061] 3061 innerTail435_6 best = best := by
  rw [innerTail435_6, scan, if_neg (by decide), innerCandidates435_6_checked]
  change scan edge (search edge 3) 4 [3061] 3061 innerTail435_7
    (if 3 ≤ innerCandidates435_6.length then search edge 3 [3061, 6343] innerCandidates435_6 9404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan435_7

def innerTail435_5 : List Nat := 5857 :: innerTail435_6
def innerCandidates435_5 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates435_5_checked : candidates edge 4 3061 5857 innerTail435_6 best.bound = innerCandidates435_5 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan435_5 : scan edge (search edge 3) 4 [3061] 3061 innerTail435_5 best = best := by
  rw [innerTail435_5, scan, if_neg (by decide), innerCandidates435_5_checked]
  change scan edge (search edge 3) 4 [3061] 3061 innerTail435_6
    (if 3 ≤ innerCandidates435_5.length then search edge 3 [3061, 5857] innerCandidates435_5 8918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan435_6

def innerTail435_4 : List Nat := 5437 :: innerTail435_5
def innerCandidates435_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates435_4_checked : candidates edge 4 3061 5437 innerTail435_5 best.bound = innerCandidates435_4 := by
  unfold candidates
  have row : edge 5437 = SearchData.row5437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan435_4 : scan edge (search edge 3) 4 [3061] 3061 innerTail435_4 best = best := by
  rw [innerTail435_4, scan, if_neg (by decide), innerCandidates435_4_checked]
  change scan edge (search edge 3) 4 [3061] 3061 innerTail435_5
    (if 3 ≤ innerCandidates435_4.length then search edge 3 [3061, 5437] innerCandidates435_4 8498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan435_5

def innerTail435_3 : List Nat := 4729 :: innerTail435_4
def innerCandidates435_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates435_3_checked : candidates edge 4 3061 4729 innerTail435_4 best.bound = innerCandidates435_3 := by
  unfold candidates
  have row : edge 4729 = SearchData.row4729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan435_3 : scan edge (search edge 3) 4 [3061] 3061 innerTail435_3 best = best := by
  rw [innerTail435_3, scan, if_neg (by decide), innerCandidates435_3_checked]
  change scan edge (search edge 3) 4 [3061] 3061 innerTail435_4
    (if 3 ≤ innerCandidates435_3.length then search edge 3 [3061, 4729] innerCandidates435_3 7790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan435_4

def innerTail435_2 : List Nat := 3613 :: innerTail435_3
def innerCandidates435_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates435_2_checked : candidates edge 4 3061 3613 innerTail435_3 best.bound = innerCandidates435_2 := by
  unfold candidates
  have row : edge 3613 = SearchData.row3613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan435_2 : scan edge (search edge 3) 4 [3061] 3061 innerTail435_2 best = best := by
  rw [innerTail435_2, scan, if_neg (by decide), innerCandidates435_2_checked]
  change scan edge (search edge 3) 4 [3061] 3061 innerTail435_3
    (if 3 ≤ innerCandidates435_2.length then search edge 3 [3061, 3613] innerCandidates435_2 6674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan435_3

def innerTail435_1 : List Nat := 3607 :: innerTail435_2
def innerCandidates435_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates435_1_checked : candidates edge 4 3061 3607 innerTail435_2 best.bound = innerCandidates435_1 := by
  unfold candidates
  have row : edge 3607 = SearchData.row3607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan435_1 : scan edge (search edge 3) 4 [3061] 3061 innerTail435_1 best = best := by
  rw [innerTail435_1, scan, if_neg (by decide), innerCandidates435_1_checked]
  change scan edge (search edge 3) 4 [3061] 3061 innerTail435_2
    (if 3 ≤ innerCandidates435_1.length then search edge 3 [3061, 3607] innerCandidates435_1 6668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan435_2

def innerTail435_0 : List Nat := 3571 :: innerTail435_1
def innerCandidates435_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates435_0_checked : candidates edge 4 3061 3571 innerTail435_1 best.bound = innerCandidates435_0 := by
  unfold candidates
  have row : edge 3571 = SearchData.row3571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan435_0 : scan edge (search edge 3) 4 [3061] 3061 innerTail435_0 best = best := by
  rw [innerTail435_0, scan, if_neg (by decide), innerCandidates435_0_checked]
  change scan edge (search edge 3) 4 [3061] 3061 innerTail435_1
    (if 3 ≤ innerCandidates435_0.length then search edge 3 [3061, 3571] innerCandidates435_0 6632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan435_1

theorem rootBranch435 : search edge 4 [3061] rootCandidates435 3061 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3061] 3061 innerTail435_0 best = best
  exact innerScan435_0

def rootCandidates434 : List Nat := [3259, 4153, 4831, 5167, 5581, 6379, 7177, 7297, 8089, 8191]
set_option maxRecDepth 4096 in
theorem rootCandidates434_checked : candidates edge 5 0 3049 tail435 best.bound = rootCandidates434 := by
  unfold candidates
  have row : edge 3049 = SearchData.row3049.contains := by funext q; rfl
  rw [row]
  decide

def innerTail434_10 : List Nat := []
theorem innerScan434_10 : scan edge (search edge 3) 4 [3049] 3049 innerTail434_10 best = best := rfl

def innerTail434_9 : List Nat := 8191 :: innerTail434_10
def innerCandidates434_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates434_9_checked : candidates edge 4 3049 8191 innerTail434_10 best.bound = innerCandidates434_9 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan434_9 : scan edge (search edge 3) 4 [3049] 3049 innerTail434_9 best = best := by
  rw [innerTail434_9, scan, if_neg (by decide), innerCandidates434_9_checked]
  change scan edge (search edge 3) 4 [3049] 3049 innerTail434_10
    (if 3 ≤ innerCandidates434_9.length then search edge 3 [3049, 8191] innerCandidates434_9 11240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan434_10

def innerTail434_8 : List Nat := 8089 :: innerTail434_9
def innerCandidates434_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates434_8_checked : candidates edge 4 3049 8089 innerTail434_9 best.bound = innerCandidates434_8 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan434_8 : scan edge (search edge 3) 4 [3049] 3049 innerTail434_8 best = best := by
  rw [innerTail434_8, scan, if_neg (by decide), innerCandidates434_8_checked]
  change scan edge (search edge 3) 4 [3049] 3049 innerTail434_9
    (if 3 ≤ innerCandidates434_8.length then search edge 3 [3049, 8089] innerCandidates434_8 11138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan434_9

def innerTail434_7 : List Nat := 7297 :: innerTail434_8
def innerCandidates434_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates434_7_checked : candidates edge 4 3049 7297 innerTail434_8 best.bound = innerCandidates434_7 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan434_7 : scan edge (search edge 3) 4 [3049] 3049 innerTail434_7 best = best := by
  rw [innerTail434_7, scan, if_neg (by decide), innerCandidates434_7_checked]
  change scan edge (search edge 3) 4 [3049] 3049 innerTail434_8
    (if 3 ≤ innerCandidates434_7.length then search edge 3 [3049, 7297] innerCandidates434_7 10346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan434_8

def innerTail434_6 : List Nat := 7177 :: innerTail434_7
def innerCandidates434_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates434_6_checked : candidates edge 4 3049 7177 innerTail434_7 best.bound = innerCandidates434_6 := by
  unfold candidates
  have row : edge 7177 = SearchData.row7177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan434_6 : scan edge (search edge 3) 4 [3049] 3049 innerTail434_6 best = best := by
  rw [innerTail434_6, scan, if_neg (by decide), innerCandidates434_6_checked]
  change scan edge (search edge 3) 4 [3049] 3049 innerTail434_7
    (if 3 ≤ innerCandidates434_6.length then search edge 3 [3049, 7177] innerCandidates434_6 10226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan434_7

def innerTail434_5 : List Nat := 6379 :: innerTail434_6
def innerCandidates434_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates434_5_checked : candidates edge 4 3049 6379 innerTail434_6 best.bound = innerCandidates434_5 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan434_5 : scan edge (search edge 3) 4 [3049] 3049 innerTail434_5 best = best := by
  rw [innerTail434_5, scan, if_neg (by decide), innerCandidates434_5_checked]
  change scan edge (search edge 3) 4 [3049] 3049 innerTail434_6
    (if 3 ≤ innerCandidates434_5.length then search edge 3 [3049, 6379] innerCandidates434_5 9428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan434_6

def innerTail434_4 : List Nat := 5581 :: innerTail434_5
def innerCandidates434_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates434_4_checked : candidates edge 4 3049 5581 innerTail434_5 best.bound = innerCandidates434_4 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan434_4 : scan edge (search edge 3) 4 [3049] 3049 innerTail434_4 best = best := by
  rw [innerTail434_4, scan, if_neg (by decide), innerCandidates434_4_checked]
  change scan edge (search edge 3) 4 [3049] 3049 innerTail434_5
    (if 3 ≤ innerCandidates434_4.length then search edge 3 [3049, 5581] innerCandidates434_4 8630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan434_5

def innerTail434_3 : List Nat := 5167 :: innerTail434_4
def innerCandidates434_3 : List Nat := [7177]
set_option maxRecDepth 4096 in
theorem innerCandidates434_3_checked : candidates edge 4 3049 5167 innerTail434_4 best.bound = innerCandidates434_3 := by
  unfold candidates
  have row : edge 5167 = SearchData.row5167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan434_3 : scan edge (search edge 3) 4 [3049] 3049 innerTail434_3 best = best := by
  rw [innerTail434_3, scan, if_neg (by decide), innerCandidates434_3_checked]
  change scan edge (search edge 3) 4 [3049] 3049 innerTail434_4
    (if 3 ≤ innerCandidates434_3.length then search edge 3 [3049, 5167] innerCandidates434_3 8216 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan434_4

def innerTail434_2 : List Nat := 4831 :: innerTail434_3
def innerCandidates434_2 : List Nat := [8191]
set_option maxRecDepth 4096 in
theorem innerCandidates434_2_checked : candidates edge 4 3049 4831 innerTail434_3 best.bound = innerCandidates434_2 := by
  unfold candidates
  have row : edge 4831 = SearchData.row4831.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan434_2 : scan edge (search edge 3) 4 [3049] 3049 innerTail434_2 best = best := by
  rw [innerTail434_2, scan, if_neg (by decide), innerCandidates434_2_checked]
  change scan edge (search edge 3) 4 [3049] 3049 innerTail434_3
    (if 3 ≤ innerCandidates434_2.length then search edge 3 [3049, 4831] innerCandidates434_2 7880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan434_3

def innerTail434_1 : List Nat := 4153 :: innerTail434_2
def innerCandidates434_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates434_1_checked : candidates edge 4 3049 4153 innerTail434_2 best.bound = innerCandidates434_1 := by
  unfold candidates
  have row : edge 4153 = SearchData.row4153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan434_1 : scan edge (search edge 3) 4 [3049] 3049 innerTail434_1 best = best := by
  rw [innerTail434_1, scan, if_neg (by decide), innerCandidates434_1_checked]
  change scan edge (search edge 3) 4 [3049] 3049 innerTail434_2
    (if 3 ≤ innerCandidates434_1.length then search edge 3 [3049, 4153] innerCandidates434_1 7202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan434_2

def innerTail434_0 : List Nat := 3259 :: innerTail434_1
def innerCandidates434_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates434_0_checked : candidates edge 4 3049 3259 innerTail434_1 best.bound = innerCandidates434_0 := by
  unfold candidates
  have row : edge 3259 = SearchData.row3259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan434_0 : scan edge (search edge 3) 4 [3049] 3049 innerTail434_0 best = best := by
  rw [innerTail434_0, scan, if_neg (by decide), innerCandidates434_0_checked]
  change scan edge (search edge 3) 4 [3049] 3049 innerTail434_1
    (if 3 ≤ innerCandidates434_0.length then search edge 3 [3049, 3259] innerCandidates434_0 6308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan434_1

theorem rootBranch434 : search edge 4 [3049] rootCandidates434 3049 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3049] 3049 innerTail434_0 best = best
  exact innerScan434_0

end ProjectEulerSolutions.PrimePairUpper.Search
