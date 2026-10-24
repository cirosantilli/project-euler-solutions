import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates124 : List Nat := [823, 967, 1471, 1543, 1663, 1933, 1951, 2647, 2731, 2749, 3067, 3109, 3181, 3469, 3769, 4177, 4519, 4597, 5521, 6079, 7213, 7459, 8209, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates124_checked : candidates edge 5 0 709 tail125 best.bound = rootCandidates124 := by
  unfold candidates
  have row : edge 709 = PrimePairCache.smallRow709.contains := by funext q; rfl
  rw [row]
  decide

def innerTail124_24 : List Nat := []
theorem innerScan124_24 : scan edge (search edge 3) 4 [709] 709 innerTail124_24 best = best := rfl

def innerTail124_23 : List Nat := 8263 :: innerTail124_24
def innerCandidates124_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_23_checked : candidates edge 4 709 8263 innerTail124_24 best.bound = innerCandidates124_23 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_23 : scan edge (search edge 3) 4 [709] 709 innerTail124_23 best = best := by
  rw [innerTail124_23, scan, if_neg (by decide), innerCandidates124_23_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_24
    (if 3 ≤ innerCandidates124_23.length then search edge 3 [709, 8263] innerCandidates124_23 8972 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_24

def innerTail124_22 : List Nat := 8209 :: innerTail124_23
def innerCandidates124_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_22_checked : candidates edge 4 709 8209 innerTail124_23 best.bound = innerCandidates124_22 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_22 : scan edge (search edge 3) 4 [709] 709 innerTail124_22 best = best := by
  rw [innerTail124_22, scan, if_neg (by decide), innerCandidates124_22_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_23
    (if 3 ≤ innerCandidates124_22.length then search edge 3 [709, 8209] innerCandidates124_22 8918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_23

def innerTail124_21 : List Nat := 7459 :: innerTail124_22
def innerCandidates124_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_21_checked : candidates edge 4 709 7459 innerTail124_22 best.bound = innerCandidates124_21 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_21 : scan edge (search edge 3) 4 [709] 709 innerTail124_21 best = best := by
  rw [innerTail124_21, scan, if_neg (by decide), innerCandidates124_21_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_22
    (if 3 ≤ innerCandidates124_21.length then search edge 3 [709, 7459] innerCandidates124_21 8168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_22

def innerTail124_20 : List Nat := 7213 :: innerTail124_21
def innerCandidates124_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_20_checked : candidates edge 4 709 7213 innerTail124_21 best.bound = innerCandidates124_20 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_20 : scan edge (search edge 3) 4 [709] 709 innerTail124_20 best = best := by
  rw [innerTail124_20, scan, if_neg (by decide), innerCandidates124_20_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_21
    (if 3 ≤ innerCandidates124_20.length then search edge 3 [709, 7213] innerCandidates124_20 7922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_21

def innerTail124_19 : List Nat := 6079 :: innerTail124_20
def innerCandidates124_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_19_checked : candidates edge 4 709 6079 innerTail124_20 best.bound = innerCandidates124_19 := by
  unfold candidates
  have row : edge 6079 = SearchData.row6079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_19 : scan edge (search edge 3) 4 [709] 709 innerTail124_19 best = best := by
  rw [innerTail124_19, scan, if_neg (by decide), innerCandidates124_19_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_20
    (if 3 ≤ innerCandidates124_19.length then search edge 3 [709, 6079] innerCandidates124_19 6788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_20

def innerTail124_18 : List Nat := 5521 :: innerTail124_19
def innerCandidates124_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_18_checked : candidates edge 4 709 5521 innerTail124_19 best.bound = innerCandidates124_18 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_18 : scan edge (search edge 3) 4 [709] 709 innerTail124_18 best = best := by
  rw [innerTail124_18, scan, if_neg (by decide), innerCandidates124_18_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_19
    (if 3 ≤ innerCandidates124_18.length then search edge 3 [709, 5521] innerCandidates124_18 6230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_19

def innerTail124_17 : List Nat := 4597 :: innerTail124_18
def innerCandidates124_17 : List Nat := [6079, 7213, 8263]
set_option maxRecDepth 4096 in
theorem innerCandidates124_17_checked : candidates edge 4 709 4597 innerTail124_18 best.bound = innerCandidates124_17 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild124_17 : search edge 3 [709, 4597] innerCandidates124_17 5306 best = best := by decide

theorem innerScan124_17 : scan edge (search edge 3) 4 [709] 709 innerTail124_17 best = best := by
  rw [innerTail124_17, scan, if_neg (by decide), innerCandidates124_17_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_18
    (if 3 ≤ innerCandidates124_17.length then search edge 3 [709, 4597] innerCandidates124_17 5306 best else best) = best
  rw [if_pos (by decide), innerChild124_17]
  exact innerScan124_18

def innerTail124_16 : List Nat := 4519 :: innerTail124_17
def innerCandidates124_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_16_checked : candidates edge 4 709 4519 innerTail124_17 best.bound = innerCandidates124_16 := by
  unfold candidates
  have row : edge 4519 = SearchData.row4519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_16 : scan edge (search edge 3) 4 [709] 709 innerTail124_16 best = best := by
  rw [innerTail124_16, scan, if_neg (by decide), innerCandidates124_16_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_17
    (if 3 ≤ innerCandidates124_16.length then search edge 3 [709, 4519] innerCandidates124_16 5228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_17

def innerTail124_15 : List Nat := 4177 :: innerTail124_16
def innerCandidates124_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_15_checked : candidates edge 4 709 4177 innerTail124_16 best.bound = innerCandidates124_15 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_15 : scan edge (search edge 3) 4 [709] 709 innerTail124_15 best = best := by
  rw [innerTail124_15, scan, if_neg (by decide), innerCandidates124_15_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_16
    (if 3 ≤ innerCandidates124_15.length then search edge 3 [709, 4177] innerCandidates124_15 4886 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_16

def innerTail124_14 : List Nat := 3769 :: innerTail124_15
def innerCandidates124_14 : List Nat := [4597]
set_option maxRecDepth 4096 in
theorem innerCandidates124_14_checked : candidates edge 4 709 3769 innerTail124_15 best.bound = innerCandidates124_14 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_14 : scan edge (search edge 3) 4 [709] 709 innerTail124_14 best = best := by
  rw [innerTail124_14, scan, if_neg (by decide), innerCandidates124_14_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_15
    (if 3 ≤ innerCandidates124_14.length then search edge 3 [709, 3769] innerCandidates124_14 4478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_15

def innerTail124_13 : List Nat := 3469 :: innerTail124_14
def innerCandidates124_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_13_checked : candidates edge 4 709 3469 innerTail124_14 best.bound = innerCandidates124_13 := by
  unfold candidates
  have row : edge 3469 = SearchData.row3469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_13 : scan edge (search edge 3) 4 [709] 709 innerTail124_13 best = best := by
  rw [innerTail124_13, scan, if_neg (by decide), innerCandidates124_13_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_14
    (if 3 ≤ innerCandidates124_13.length then search edge 3 [709, 3469] innerCandidates124_13 4178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_14

def innerTail124_12 : List Nat := 3181 :: innerTail124_13
def innerCandidates124_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_12_checked : candidates edge 4 709 3181 innerTail124_13 best.bound = innerCandidates124_12 := by
  unfold candidates
  have row : edge 3181 = SearchData.row3181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_12 : scan edge (search edge 3) 4 [709] 709 innerTail124_12 best = best := by
  rw [innerTail124_12, scan, if_neg (by decide), innerCandidates124_12_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_13
    (if 3 ≤ innerCandidates124_12.length then search edge 3 [709, 3181] innerCandidates124_12 3890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_13

def innerTail124_11 : List Nat := 3109 :: innerTail124_12
def innerCandidates124_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_11_checked : candidates edge 4 709 3109 innerTail124_12 best.bound = innerCandidates124_11 := by
  unfold candidates
  have row : edge 3109 = SearchData.row3109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_11 : scan edge (search edge 3) 4 [709] 709 innerTail124_11 best = best := by
  rw [innerTail124_11, scan, if_neg (by decide), innerCandidates124_11_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_12
    (if 3 ≤ innerCandidates124_11.length then search edge 3 [709, 3109] innerCandidates124_11 3818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_12

def innerTail124_10 : List Nat := 3067 :: innerTail124_11
def innerCandidates124_10 : List Nat := [7459]
set_option maxRecDepth 4096 in
theorem innerCandidates124_10_checked : candidates edge 4 709 3067 innerTail124_11 best.bound = innerCandidates124_10 := by
  unfold candidates
  have row : edge 3067 = SearchData.row3067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_10 : scan edge (search edge 3) 4 [709] 709 innerTail124_10 best = best := by
  rw [innerTail124_10, scan, if_neg (by decide), innerCandidates124_10_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_11
    (if 3 ≤ innerCandidates124_10.length then search edge 3 [709, 3067] innerCandidates124_10 3776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_11

def innerTail124_9 : List Nat := 2749 :: innerTail124_10
def innerCandidates124_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_9_checked : candidates edge 4 709 2749 innerTail124_10 best.bound = innerCandidates124_9 := by
  unfold candidates
  have row : edge 2749 = SearchData.row2749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_9 : scan edge (search edge 3) 4 [709] 709 innerTail124_9 best = best := by
  rw [innerTail124_9, scan, if_neg (by decide), innerCandidates124_9_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_10
    (if 3 ≤ innerCandidates124_9.length then search edge 3 [709, 2749] innerCandidates124_9 3458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_10

def innerTail124_8 : List Nat := 2731 :: innerTail124_9
def innerCandidates124_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_8_checked : candidates edge 4 709 2731 innerTail124_9 best.bound = innerCandidates124_8 := by
  unfold candidates
  have row : edge 2731 = SearchData.row2731.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_8 : scan edge (search edge 3) 4 [709] 709 innerTail124_8 best = best := by
  rw [innerTail124_8, scan, if_neg (by decide), innerCandidates124_8_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_9
    (if 3 ≤ innerCandidates124_8.length then search edge 3 [709, 2731] innerCandidates124_8 3440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_9

def innerTail124_7 : List Nat := 2647 :: innerTail124_8
def innerCandidates124_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_7_checked : candidates edge 4 709 2647 innerTail124_8 best.bound = innerCandidates124_7 := by
  unfold candidates
  have row : edge 2647 = PrimePairCache.smallRow2647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_7 : scan edge (search edge 3) 4 [709] 709 innerTail124_7 best = best := by
  rw [innerTail124_7, scan, if_neg (by decide), innerCandidates124_7_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_8
    (if 3 ≤ innerCandidates124_7.length then search edge 3 [709, 2647] innerCandidates124_7 3356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_8

def innerTail124_6 : List Nat := 1951 :: innerTail124_7
def innerCandidates124_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_6_checked : candidates edge 4 709 1951 innerTail124_7 best.bound = innerCandidates124_6 := by
  unfold candidates
  have row : edge 1951 = PrimePairCache.smallRow1951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_6 : scan edge (search edge 3) 4 [709] 709 innerTail124_6 best = best := by
  rw [innerTail124_6, scan, if_neg (by decide), innerCandidates124_6_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_7
    (if 3 ≤ innerCandidates124_6.length then search edge 3 [709, 1951] innerCandidates124_6 2660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_7

def innerTail124_5 : List Nat := 1933 :: innerTail124_6
def innerCandidates124_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_5_checked : candidates edge 4 709 1933 innerTail124_6 best.bound = innerCandidates124_5 := by
  unfold candidates
  have row : edge 1933 = PrimePairCache.smallRow1933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_5 : scan edge (search edge 3) 4 [709] 709 innerTail124_5 best = best := by
  rw [innerTail124_5, scan, if_neg (by decide), innerCandidates124_5_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_6
    (if 3 ≤ innerCandidates124_5.length then search edge 3 [709, 1933] innerCandidates124_5 2642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_6

def innerTail124_4 : List Nat := 1663 :: innerTail124_5
def innerCandidates124_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_4_checked : candidates edge 4 709 1663 innerTail124_5 best.bound = innerCandidates124_4 := by
  unfold candidates
  have row : edge 1663 = PrimePairCache.smallRow1663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_4 : scan edge (search edge 3) 4 [709] 709 innerTail124_4 best = best := by
  rw [innerTail124_4, scan, if_neg (by decide), innerCandidates124_4_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_5
    (if 3 ≤ innerCandidates124_4.length then search edge 3 [709, 1663] innerCandidates124_4 2372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_5

def innerTail124_3 : List Nat := 1543 :: innerTail124_4
def innerCandidates124_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_3_checked : candidates edge 4 709 1543 innerTail124_4 best.bound = innerCandidates124_3 := by
  unfold candidates
  have row : edge 1543 = PrimePairCache.smallRow1543.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_3 : scan edge (search edge 3) 4 [709] 709 innerTail124_3 best = best := by
  rw [innerTail124_3, scan, if_neg (by decide), innerCandidates124_3_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_4
    (if 3 ≤ innerCandidates124_3.length then search edge 3 [709, 1543] innerCandidates124_3 2252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_4

def innerTail124_2 : List Nat := 1471 :: innerTail124_3
def innerCandidates124_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates124_2_checked : candidates edge 4 709 1471 innerTail124_3 best.bound = innerCandidates124_2 := by
  unfold candidates
  have row : edge 1471 = PrimePairCache.smallRow1471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_2 : scan edge (search edge 3) 4 [709] 709 innerTail124_2 best = best := by
  rw [innerTail124_2, scan, if_neg (by decide), innerCandidates124_2_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_3
    (if 3 ≤ innerCandidates124_2.length then search edge 3 [709, 1471] innerCandidates124_2 2180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_3

def innerTail124_1 : List Nat := 967 :: innerTail124_2
def innerCandidates124_1 : List Nat := [1663, 1951]
set_option maxRecDepth 4096 in
theorem innerCandidates124_1_checked : candidates edge 4 709 967 innerTail124_2 best.bound = innerCandidates124_1 := by
  unfold candidates
  have row : edge 967 = PrimePairCache.smallRow967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_1 : scan edge (search edge 3) 4 [709] 709 innerTail124_1 best = best := by
  rw [innerTail124_1, scan, if_neg (by decide), innerCandidates124_1_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_2
    (if 3 ≤ innerCandidates124_1.length then search edge 3 [709, 967] innerCandidates124_1 1676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_2

def innerTail124_0 : List Nat := 823 :: innerTail124_1
def innerCandidates124_0 : List Nat := [967, 1933]
set_option maxRecDepth 4096 in
theorem innerCandidates124_0_checked : candidates edge 4 709 823 innerTail124_1 best.bound = innerCandidates124_0 := by
  unfold candidates
  have row : edge 823 = PrimePairCache.smallRow823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan124_0 : scan edge (search edge 3) 4 [709] 709 innerTail124_0 best = best := by
  rw [innerTail124_0, scan, if_neg (by decide), innerCandidates124_0_checked]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_1
    (if 3 ≤ innerCandidates124_0.length then search edge 3 [709, 823] innerCandidates124_0 1532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan124_1

theorem rootBranch124 : search edge 4 [709] rootCandidates124 709 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [709] 709 innerTail124_0 best = best
  exact innerScan124_0

def rootCandidates123 : List Nat := [1181, 1187, 1283, 1307, 1439, 1553, 1607, 1733, 1913, 2267, 2477, 3581, 3623, 4283, 4373, 4583, 4751, 4877, 5003, 5087, 5417, 5471, 5501, 6101, 6389, 6803, 7541, 7607, 7919, 7937, 8171]
set_option maxRecDepth 4096 in
theorem rootCandidates123_checked : candidates edge 5 0 701 tail124 best.bound = rootCandidates123 := by
  unfold candidates
  have row : edge 701 = PrimePairCache.smallRow701.contains := by funext q; rfl
  rw [row]
  decide

def innerTail123_31 : List Nat := []
theorem innerScan123_31 : scan edge (search edge 3) 4 [701] 701 innerTail123_31 best = best := rfl

def innerTail123_30 : List Nat := 8171 :: innerTail123_31
def innerCandidates123_30 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates123_30_checked : candidates edge 4 701 8171 innerTail123_31 best.bound = innerCandidates123_30 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_30 : scan edge (search edge 3) 4 [701] 701 innerTail123_30 best = best := by
  rw [innerTail123_30, scan, if_neg (by decide), innerCandidates123_30_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_31
    (if 3 ≤ innerCandidates123_30.length then search edge 3 [701, 8171] innerCandidates123_30 8872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_31

def innerTail123_29 : List Nat := 7937 :: innerTail123_30
def innerCandidates123_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates123_29_checked : candidates edge 4 701 7937 innerTail123_30 best.bound = innerCandidates123_29 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_29 : scan edge (search edge 3) 4 [701] 701 innerTail123_29 best = best := by
  rw [innerTail123_29, scan, if_neg (by decide), innerCandidates123_29_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_30
    (if 3 ≤ innerCandidates123_29.length then search edge 3 [701, 7937] innerCandidates123_29 8638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_30

def innerTail123_28 : List Nat := 7919 :: innerTail123_29
def innerCandidates123_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates123_28_checked : candidates edge 4 701 7919 innerTail123_29 best.bound = innerCandidates123_28 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_28 : scan edge (search edge 3) 4 [701] 701 innerTail123_28 best = best := by
  rw [innerTail123_28, scan, if_neg (by decide), innerCandidates123_28_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_29
    (if 3 ≤ innerCandidates123_28.length then search edge 3 [701, 7919] innerCandidates123_28 8620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_29

def innerTail123_27 : List Nat := 7607 :: innerTail123_28
def innerCandidates123_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates123_27_checked : candidates edge 4 701 7607 innerTail123_28 best.bound = innerCandidates123_27 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_27 : scan edge (search edge 3) 4 [701] 701 innerTail123_27 best = best := by
  rw [innerTail123_27, scan, if_neg (by decide), innerCandidates123_27_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_28
    (if 3 ≤ innerCandidates123_27.length then search edge 3 [701, 7607] innerCandidates123_27 8308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_28

def innerTail123_26 : List Nat := 7541 :: innerTail123_27
def innerCandidates123_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates123_26_checked : candidates edge 4 701 7541 innerTail123_27 best.bound = innerCandidates123_26 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_26 : scan edge (search edge 3) 4 [701] 701 innerTail123_26 best = best := by
  rw [innerTail123_26, scan, if_neg (by decide), innerCandidates123_26_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_27
    (if 3 ≤ innerCandidates123_26.length then search edge 3 [701, 7541] innerCandidates123_26 8242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_27

def innerTail123_25 : List Nat := 6803 :: innerTail123_26
def innerCandidates123_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates123_25_checked : candidates edge 4 701 6803 innerTail123_26 best.bound = innerCandidates123_25 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_25 : scan edge (search edge 3) 4 [701] 701 innerTail123_25 best = best := by
  rw [innerTail123_25, scan, if_neg (by decide), innerCandidates123_25_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_26
    (if 3 ≤ innerCandidates123_25.length then search edge 3 [701, 6803] innerCandidates123_25 7504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_26

def innerTail123_24 : List Nat := 6389 :: innerTail123_25
def innerCandidates123_24 : List Nat := [7937]
set_option maxRecDepth 4096 in
theorem innerCandidates123_24_checked : candidates edge 4 701 6389 innerTail123_25 best.bound = innerCandidates123_24 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_24 : scan edge (search edge 3) 4 [701] 701 innerTail123_24 best = best := by
  rw [innerTail123_24, scan, if_neg (by decide), innerCandidates123_24_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_25
    (if 3 ≤ innerCandidates123_24.length then search edge 3 [701, 6389] innerCandidates123_24 7090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_25

def innerTail123_23 : List Nat := 6101 :: innerTail123_24
def innerCandidates123_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates123_23_checked : candidates edge 4 701 6101 innerTail123_24 best.bound = innerCandidates123_23 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_23 : scan edge (search edge 3) 4 [701] 701 innerTail123_23 best = best := by
  rw [innerTail123_23, scan, if_neg (by decide), innerCandidates123_23_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_24
    (if 3 ≤ innerCandidates123_23.length then search edge 3 [701, 6101] innerCandidates123_23 6802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_24

def innerTail123_22 : List Nat := 5501 :: innerTail123_23
def innerCandidates123_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates123_22_checked : candidates edge 4 701 5501 innerTail123_23 best.bound = innerCandidates123_22 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_22 : scan edge (search edge 3) 4 [701] 701 innerTail123_22 best = best := by
  rw [innerTail123_22, scan, if_neg (by decide), innerCandidates123_22_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_23
    (if 3 ≤ innerCandidates123_22.length then search edge 3 [701, 5501] innerCandidates123_22 6202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_23

def innerTail123_21 : List Nat := 5471 :: innerTail123_22
def innerCandidates123_21 : List Nat := [8171]
set_option maxRecDepth 4096 in
theorem innerCandidates123_21_checked : candidates edge 4 701 5471 innerTail123_22 best.bound = innerCandidates123_21 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_21 : scan edge (search edge 3) 4 [701] 701 innerTail123_21 best = best := by
  rw [innerTail123_21, scan, if_neg (by decide), innerCandidates123_21_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_22
    (if 3 ≤ innerCandidates123_21.length then search edge 3 [701, 5471] innerCandidates123_21 6172 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_22

def innerTail123_20 : List Nat := 5417 :: innerTail123_21
def innerCandidates123_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates123_20_checked : candidates edge 4 701 5417 innerTail123_21 best.bound = innerCandidates123_20 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_20 : scan edge (search edge 3) 4 [701] 701 innerTail123_20 best = best := by
  rw [innerTail123_20, scan, if_neg (by decide), innerCandidates123_20_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_21
    (if 3 ≤ innerCandidates123_20.length then search edge 3 [701, 5417] innerCandidates123_20 6118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_21

def innerTail123_19 : List Nat := 5087 :: innerTail123_20
def innerCandidates123_19 : List Nat := [6803]
set_option maxRecDepth 4096 in
theorem innerCandidates123_19_checked : candidates edge 4 701 5087 innerTail123_20 best.bound = innerCandidates123_19 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_19 : scan edge (search edge 3) 4 [701] 701 innerTail123_19 best = best := by
  rw [innerTail123_19, scan, if_neg (by decide), innerCandidates123_19_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_20
    (if 3 ≤ innerCandidates123_19.length then search edge 3 [701, 5087] innerCandidates123_19 5788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_20

def innerTail123_18 : List Nat := 5003 :: innerTail123_19
def innerCandidates123_18 : List Nat := [5087, 7541]
set_option maxRecDepth 4096 in
theorem innerCandidates123_18_checked : candidates edge 4 701 5003 innerTail123_19 best.bound = innerCandidates123_18 := by
  unfold candidates
  have row : edge 5003 = SearchData.row5003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_18 : scan edge (search edge 3) 4 [701] 701 innerTail123_18 best = best := by
  rw [innerTail123_18, scan, if_neg (by decide), innerCandidates123_18_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_19
    (if 3 ≤ innerCandidates123_18.length then search edge 3 [701, 5003] innerCandidates123_18 5704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_19

def innerTail123_17 : List Nat := 4877 :: innerTail123_18
def innerCandidates123_17 : List Nat := [5501]
set_option maxRecDepth 4096 in
theorem innerCandidates123_17_checked : candidates edge 4 701 4877 innerTail123_18 best.bound = innerCandidates123_17 := by
  unfold candidates
  have row : edge 4877 = SearchData.row4877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_17 : scan edge (search edge 3) 4 [701] 701 innerTail123_17 best = best := by
  rw [innerTail123_17, scan, if_neg (by decide), innerCandidates123_17_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_18
    (if 3 ≤ innerCandidates123_17.length then search edge 3 [701, 4877] innerCandidates123_17 5578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_18

def innerTail123_16 : List Nat := 4751 :: innerTail123_17
def innerCandidates123_16 : List Nat := [7607]
set_option maxRecDepth 4096 in
theorem innerCandidates123_16_checked : candidates edge 4 701 4751 innerTail123_17 best.bound = innerCandidates123_16 := by
  unfold candidates
  have row : edge 4751 = SearchData.row4751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_16 : scan edge (search edge 3) 4 [701] 701 innerTail123_16 best = best := by
  rw [innerTail123_16, scan, if_neg (by decide), innerCandidates123_16_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_17
    (if 3 ≤ innerCandidates123_16.length then search edge 3 [701, 4751] innerCandidates123_16 5452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_17

def innerTail123_15 : List Nat := 4583 :: innerTail123_16
def innerCandidates123_15 : List Nat := [5003]
set_option maxRecDepth 4096 in
theorem innerCandidates123_15_checked : candidates edge 4 701 4583 innerTail123_16 best.bound = innerCandidates123_15 := by
  unfold candidates
  have row : edge 4583 = SearchData.row4583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_15 : scan edge (search edge 3) 4 [701] 701 innerTail123_15 best = best := by
  rw [innerTail123_15, scan, if_neg (by decide), innerCandidates123_15_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_16
    (if 3 ≤ innerCandidates123_15.length then search edge 3 [701, 4583] innerCandidates123_15 5284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_16

def innerTail123_14 : List Nat := 4373 :: innerTail123_15
def innerCandidates123_14 : List Nat := [7541, 7919]
set_option maxRecDepth 4096 in
theorem innerCandidates123_14_checked : candidates edge 4 701 4373 innerTail123_15 best.bound = innerCandidates123_14 := by
  unfold candidates
  have row : edge 4373 = SearchData.row4373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_14 : scan edge (search edge 3) 4 [701] 701 innerTail123_14 best = best := by
  rw [innerTail123_14, scan, if_neg (by decide), innerCandidates123_14_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_15
    (if 3 ≤ innerCandidates123_14.length then search edge 3 [701, 4373] innerCandidates123_14 5074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_15

def innerTail123_13 : List Nat := 4283 :: innerTail123_14
def innerCandidates123_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates123_13_checked : candidates edge 4 701 4283 innerTail123_14 best.bound = innerCandidates123_13 := by
  unfold candidates
  have row : edge 4283 = SearchData.row4283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_13 : scan edge (search edge 3) 4 [701] 701 innerTail123_13 best = best := by
  rw [innerTail123_13, scan, if_neg (by decide), innerCandidates123_13_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_14
    (if 3 ≤ innerCandidates123_13.length then search edge 3 [701, 4283] innerCandidates123_13 4984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_14

def innerTail123_12 : List Nat := 3623 :: innerTail123_13
def innerCandidates123_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates123_12_checked : candidates edge 4 701 3623 innerTail123_13 best.bound = innerCandidates123_12 := by
  unfold candidates
  have row : edge 3623 = SearchData.row3623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_12 : scan edge (search edge 3) 4 [701] 701 innerTail123_12 best = best := by
  rw [innerTail123_12, scan, if_neg (by decide), innerCandidates123_12_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_13
    (if 3 ≤ innerCandidates123_12.length then search edge 3 [701, 3623] innerCandidates123_12 4324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_13

def innerTail123_11 : List Nat := 3581 :: innerTail123_12
def innerCandidates123_11 : List Nat := [6101]
set_option maxRecDepth 4096 in
theorem innerCandidates123_11_checked : candidates edge 4 701 3581 innerTail123_12 best.bound = innerCandidates123_11 := by
  unfold candidates
  have row : edge 3581 = SearchData.row3581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_11 : scan edge (search edge 3) 4 [701] 701 innerTail123_11 best = best := by
  rw [innerTail123_11, scan, if_neg (by decide), innerCandidates123_11_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_12
    (if 3 ≤ innerCandidates123_11.length then search edge 3 [701, 3581] innerCandidates123_11 4282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_12

def innerTail123_10 : List Nat := 2477 :: innerTail123_11
def innerCandidates123_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates123_10_checked : candidates edge 4 701 2477 innerTail123_11 best.bound = innerCandidates123_10 := by
  unfold candidates
  have row : edge 2477 = PrimePairCache.smallRow2477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_10 : scan edge (search edge 3) 4 [701] 701 innerTail123_10 best = best := by
  rw [innerTail123_10, scan, if_neg (by decide), innerCandidates123_10_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_11
    (if 3 ≤ innerCandidates123_10.length then search edge 3 [701, 2477] innerCandidates123_10 3178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_11

def innerTail123_9 : List Nat := 2267 :: innerTail123_10
def innerCandidates123_9 : List Nat := [6389]
set_option maxRecDepth 4096 in
theorem innerCandidates123_9_checked : candidates edge 4 701 2267 innerTail123_10 best.bound = innerCandidates123_9 := by
  unfold candidates
  have row : edge 2267 = PrimePairCache.smallRow2267.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_9 : scan edge (search edge 3) 4 [701] 701 innerTail123_9 best = best := by
  rw [innerTail123_9, scan, if_neg (by decide), innerCandidates123_9_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_10
    (if 3 ≤ innerCandidates123_9.length then search edge 3 [701, 2267] innerCandidates123_9 2968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_10

def innerTail123_8 : List Nat := 1913 :: innerTail123_9
def innerCandidates123_8 : List Nat := [5087, 6389, 7919]
set_option maxRecDepth 4096 in
theorem innerCandidates123_8_checked : candidates edge 4 701 1913 innerTail123_9 best.bound = innerCandidates123_8 := by
  unfold candidates
  have row : edge 1913 = PrimePairCache.smallRow1913.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild123_8 : search edge 3 [701, 1913] innerCandidates123_8 2614 best = best := by decide

theorem innerScan123_8 : scan edge (search edge 3) 4 [701] 701 innerTail123_8 best = best := by
  rw [innerTail123_8, scan, if_neg (by decide), innerCandidates123_8_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_9
    (if 3 ≤ innerCandidates123_8.length then search edge 3 [701, 1913] innerCandidates123_8 2614 best else best) = best
  rw [if_pos (by decide), innerChild123_8]
  exact innerScan123_9

def innerTail123_7 : List Nat := 1733 :: innerTail123_8
def innerCandidates123_7 : List Nat := [6101]
set_option maxRecDepth 4096 in
theorem innerCandidates123_7_checked : candidates edge 4 701 1733 innerTail123_8 best.bound = innerCandidates123_7 := by
  unfold candidates
  have row : edge 1733 = PrimePairCache.smallRow1733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_7 : scan edge (search edge 3) 4 [701] 701 innerTail123_7 best = best := by
  rw [innerTail123_7, scan, if_neg (by decide), innerCandidates123_7_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_8
    (if 3 ≤ innerCandidates123_7.length then search edge 3 [701, 1733] innerCandidates123_7 2434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_8

def innerTail123_6 : List Nat := 1607 :: innerTail123_7
def innerCandidates123_6 : List Nat := [5471, 7541, 8171]
set_option maxRecDepth 4096 in
theorem innerCandidates123_6_checked : candidates edge 4 701 1607 innerTail123_7 best.bound = innerCandidates123_6 := by
  unfold candidates
  have row : edge 1607 = PrimePairCache.smallRow1607.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild123_6 : search edge 3 [701, 1607] innerCandidates123_6 2308 best = best := by decide

theorem innerScan123_6 : scan edge (search edge 3) 4 [701] 701 innerTail123_6 best = best := by
  rw [innerTail123_6, scan, if_neg (by decide), innerCandidates123_6_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_7
    (if 3 ≤ innerCandidates123_6.length then search edge 3 [701, 1607] innerCandidates123_6 2308 best else best) = best
  rw [if_pos (by decide), innerChild123_6]
  exact innerScan123_7

def innerTail123_5 : List Nat := 1553 :: innerTail123_6
def innerCandidates123_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates123_5_checked : candidates edge 4 701 1553 innerTail123_6 best.bound = innerCandidates123_5 := by
  unfold candidates
  have row : edge 1553 = PrimePairCache.smallRow1553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_5 : scan edge (search edge 3) 4 [701] 701 innerTail123_5 best = best := by
  rw [innerTail123_5, scan, if_neg (by decide), innerCandidates123_5_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_6
    (if 3 ≤ innerCandidates123_5.length then search edge 3 [701, 1553] innerCandidates123_5 2254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_6

def innerTail123_4 : List Nat := 1439 :: innerTail123_5
def innerCandidates123_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates123_4_checked : candidates edge 4 701 1439 innerTail123_5 best.bound = innerCandidates123_4 := by
  unfold candidates
  have row : edge 1439 = PrimePairCache.smallRow1439.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_4 : scan edge (search edge 3) 4 [701] 701 innerTail123_4 best = best := by
  rw [innerTail123_4, scan, if_neg (by decide), innerCandidates123_4_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_5
    (if 3 ≤ innerCandidates123_4.length then search edge 3 [701, 1439] innerCandidates123_4 2140 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_5

def innerTail123_3 : List Nat := 1307 :: innerTail123_4
def innerCandidates123_3 : List Nat := [4583, 7541, 8171]
set_option maxRecDepth 4096 in
theorem innerCandidates123_3_checked : candidates edge 4 701 1307 innerTail123_4 best.bound = innerCandidates123_3 := by
  unfold candidates
  have row : edge 1307 = PrimePairCache.smallRow1307.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild123_3 : search edge 3 [701, 1307] innerCandidates123_3 2008 best = best := by decide

theorem innerScan123_3 : scan edge (search edge 3) 4 [701] 701 innerTail123_3 best = best := by
  rw [innerTail123_3, scan, if_neg (by decide), innerCandidates123_3_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_4
    (if 3 ≤ innerCandidates123_3.length then search edge 3 [701, 1307] innerCandidates123_3 2008 best else best) = best
  rw [if_pos (by decide), innerChild123_3]
  exact innerScan123_4

def innerTail123_2 : List Nat := 1283 :: innerTail123_3
def innerCandidates123_2 : List Nat := [1913]
set_option maxRecDepth 4096 in
theorem innerCandidates123_2_checked : candidates edge 4 701 1283 innerTail123_3 best.bound = innerCandidates123_2 := by
  unfold candidates
  have row : edge 1283 = PrimePairCache.smallRow1283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_2 : scan edge (search edge 3) 4 [701] 701 innerTail123_2 best = best := by
  rw [innerTail123_2, scan, if_neg (by decide), innerCandidates123_2_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_3
    (if 3 ≤ innerCandidates123_2.length then search edge 3 [701, 1283] innerCandidates123_2 1984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_3

def innerTail123_1 : List Nat := 1187 :: innerTail123_2
def innerCandidates123_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates123_1_checked : candidates edge 4 701 1187 innerTail123_2 best.bound = innerCandidates123_1 := by
  unfold candidates
  have row : edge 1187 = PrimePairCache.smallRow1187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_1 : scan edge (search edge 3) 4 [701] 701 innerTail123_1 best = best := by
  rw [innerTail123_1, scan, if_neg (by decide), innerCandidates123_1_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_2
    (if 3 ≤ innerCandidates123_1.length then search edge 3 [701, 1187] innerCandidates123_1 1888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_2

def innerTail123_0 : List Nat := 1181 :: innerTail123_1
def innerCandidates123_0 : List Nat := [5501]
set_option maxRecDepth 4096 in
theorem innerCandidates123_0_checked : candidates edge 4 701 1181 innerTail123_1 best.bound = innerCandidates123_0 := by
  unfold candidates
  have row : edge 1181 = PrimePairCache.smallRow1181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan123_0 : scan edge (search edge 3) 4 [701] 701 innerTail123_0 best = best := by
  rw [innerTail123_0, scan, if_neg (by decide), innerCandidates123_0_checked]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_1
    (if 3 ≤ innerCandidates123_0.length then search edge 3 [701, 1181] innerCandidates123_0 1882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan123_1

theorem rootBranch123 : search edge 4 [701] rootCandidates123 701 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [701] 701 innerTail123_0 best = best
  exact innerScan123_0

end ProjectEulerSolutions.PrimePairUpper.Search
