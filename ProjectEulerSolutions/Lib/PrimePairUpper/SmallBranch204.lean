import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates94 : List Nat := [647, 797, 863, 947, 1019, 1061, 1091, 1451, 1553, 1613, 1973, 2111, 3359, 3929, 4241, 4673, 5471, 5483, 5711, 6563, 6863, 7193, 7577, 7583, 7793, 8117, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates94_checked : candidates edge 5 0 509 tail95 best.bound = rootCandidates94 := by
  unfold candidates
  have row : edge 509 = PrimePairCache.smallRow509.contains := by funext q; rfl
  rw [row]
  decide

def innerTail94_27 : List Nat := []
theorem innerScan94_27 : scan edge (search edge 3) 4 [509] 509 innerTail94_27 best = best := rfl

def innerTail94_26 : List Nat := 8369 :: innerTail94_27
def innerCandidates94_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates94_26_checked : candidates edge 4 509 8369 innerTail94_27 best.bound = innerCandidates94_26 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_26 : scan edge (search edge 3) 4 [509] 509 innerTail94_26 best = best := by
  rw [innerTail94_26, scan, if_neg (by decide), innerCandidates94_26_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_27
    (if 3 ≤ innerCandidates94_26.length then search edge 3 [509, 8369] innerCandidates94_26 8878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_27

def innerTail94_25 : List Nat := 8117 :: innerTail94_26
def innerCandidates94_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates94_25_checked : candidates edge 4 509 8117 innerTail94_26 best.bound = innerCandidates94_25 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_25 : scan edge (search edge 3) 4 [509] 509 innerTail94_25 best = best := by
  rw [innerTail94_25, scan, if_neg (by decide), innerCandidates94_25_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_26
    (if 3 ≤ innerCandidates94_25.length then search edge 3 [509, 8117] innerCandidates94_25 8626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_26

def innerTail94_24 : List Nat := 7793 :: innerTail94_25
def innerCandidates94_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates94_24_checked : candidates edge 4 509 7793 innerTail94_25 best.bound = innerCandidates94_24 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_24 : scan edge (search edge 3) 4 [509] 509 innerTail94_24 best = best := by
  rw [innerTail94_24, scan, if_neg (by decide), innerCandidates94_24_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_25
    (if 3 ≤ innerCandidates94_24.length then search edge 3 [509, 7793] innerCandidates94_24 8302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_25

def innerTail94_23 : List Nat := 7583 :: innerTail94_24
def innerCandidates94_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates94_23_checked : candidates edge 4 509 7583 innerTail94_24 best.bound = innerCandidates94_23 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_23 : scan edge (search edge 3) 4 [509] 509 innerTail94_23 best = best := by
  rw [innerTail94_23, scan, if_neg (by decide), innerCandidates94_23_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_24
    (if 3 ≤ innerCandidates94_23.length then search edge 3 [509, 7583] innerCandidates94_23 8092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_24

def innerTail94_22 : List Nat := 7577 :: innerTail94_23
def innerCandidates94_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates94_22_checked : candidates edge 4 509 7577 innerTail94_23 best.bound = innerCandidates94_22 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_22 : scan edge (search edge 3) 4 [509] 509 innerTail94_22 best = best := by
  rw [innerTail94_22, scan, if_neg (by decide), innerCandidates94_22_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_23
    (if 3 ≤ innerCandidates94_22.length then search edge 3 [509, 7577] innerCandidates94_22 8086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_23

def innerTail94_21 : List Nat := 7193 :: innerTail94_22
def innerCandidates94_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates94_21_checked : candidates edge 4 509 7193 innerTail94_22 best.bound = innerCandidates94_21 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_21 : scan edge (search edge 3) 4 [509] 509 innerTail94_21 best = best := by
  rw [innerTail94_21, scan, if_neg (by decide), innerCandidates94_21_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_22
    (if 3 ≤ innerCandidates94_21.length then search edge 3 [509, 7193] innerCandidates94_21 7702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_22

def innerTail94_20 : List Nat := 6863 :: innerTail94_21
def innerCandidates94_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates94_20_checked : candidates edge 4 509 6863 innerTail94_21 best.bound = innerCandidates94_20 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_20 : scan edge (search edge 3) 4 [509] 509 innerTail94_20 best = best := by
  rw [innerTail94_20, scan, if_neg (by decide), innerCandidates94_20_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_21
    (if 3 ≤ innerCandidates94_20.length then search edge 3 [509, 6863] innerCandidates94_20 7372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_21

def innerTail94_19 : List Nat := 6563 :: innerTail94_20
def innerCandidates94_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates94_19_checked : candidates edge 4 509 6563 innerTail94_20 best.bound = innerCandidates94_19 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_19 : scan edge (search edge 3) 4 [509] 509 innerTail94_19 best = best := by
  rw [innerTail94_19, scan, if_neg (by decide), innerCandidates94_19_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_20
    (if 3 ≤ innerCandidates94_19.length then search edge 3 [509, 6563] innerCandidates94_19 7072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_20

def innerTail94_18 : List Nat := 5711 :: innerTail94_19
def innerCandidates94_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates94_18_checked : candidates edge 4 509 5711 innerTail94_19 best.bound = innerCandidates94_18 := by
  unfold candidates
  have row : edge 5711 = SearchData.row5711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_18 : scan edge (search edge 3) 4 [509] 509 innerTail94_18 best = best := by
  rw [innerTail94_18, scan, if_neg (by decide), innerCandidates94_18_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_19
    (if 3 ≤ innerCandidates94_18.length then search edge 3 [509, 5711] innerCandidates94_18 6220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_19

def innerTail94_17 : List Nat := 5483 :: innerTail94_18
def innerCandidates94_17 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates94_17_checked : candidates edge 4 509 5483 innerTail94_18 best.bound = innerCandidates94_17 := by
  unfold candidates
  have row : edge 5483 = SearchData.row5483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_17 : scan edge (search edge 3) 4 [509] 509 innerTail94_17 best = best := by
  rw [innerTail94_17, scan, if_neg (by decide), innerCandidates94_17_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_18
    (if 3 ≤ innerCandidates94_17.length then search edge 3 [509, 5483] innerCandidates94_17 5992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_18

def innerTail94_16 : List Nat := 5471 :: innerTail94_17
def innerCandidates94_16 : List Nat := [7583]
set_option maxRecDepth 4096 in
theorem innerCandidates94_16_checked : candidates edge 4 509 5471 innerTail94_17 best.bound = innerCandidates94_16 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_16 : scan edge (search edge 3) 4 [509] 509 innerTail94_16 best = best := by
  rw [innerTail94_16, scan, if_neg (by decide), innerCandidates94_16_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_17
    (if 3 ≤ innerCandidates94_16.length then search edge 3 [509, 5471] innerCandidates94_16 5980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_17

def innerTail94_15 : List Nat := 4673 :: innerTail94_16
def innerCandidates94_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates94_15_checked : candidates edge 4 509 4673 innerTail94_16 best.bound = innerCandidates94_15 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_15 : scan edge (search edge 3) 4 [509] 509 innerTail94_15 best = best := by
  rw [innerTail94_15, scan, if_neg (by decide), innerCandidates94_15_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_16
    (if 3 ≤ innerCandidates94_15.length then search edge 3 [509, 4673] innerCandidates94_15 5182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_16

def innerTail94_14 : List Nat := 4241 :: innerTail94_15
def innerCandidates94_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates94_14_checked : candidates edge 4 509 4241 innerTail94_15 best.bound = innerCandidates94_14 := by
  unfold candidates
  have row : edge 4241 = SearchData.row4241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_14 : scan edge (search edge 3) 4 [509] 509 innerTail94_14 best = best := by
  rw [innerTail94_14, scan, if_neg (by decide), innerCandidates94_14_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_15
    (if 3 ≤ innerCandidates94_14.length then search edge 3 [509, 4241] innerCandidates94_14 4750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_15

def innerTail94_13 : List Nat := 3929 :: innerTail94_14
def innerCandidates94_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates94_13_checked : candidates edge 4 509 3929 innerTail94_14 best.bound = innerCandidates94_13 := by
  unfold candidates
  have row : edge 3929 = SearchData.row3929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_13 : scan edge (search edge 3) 4 [509] 509 innerTail94_13 best = best := by
  rw [innerTail94_13, scan, if_neg (by decide), innerCandidates94_13_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_14
    (if 3 ≤ innerCandidates94_13.length then search edge 3 [509, 3929] innerCandidates94_13 4438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_14

def innerTail94_12 : List Nat := 3359 :: innerTail94_13
def innerCandidates94_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates94_12_checked : candidates edge 4 509 3359 innerTail94_13 best.bound = innerCandidates94_12 := by
  unfold candidates
  have row : edge 3359 = SearchData.row3359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_12 : scan edge (search edge 3) 4 [509] 509 innerTail94_12 best = best := by
  rw [innerTail94_12, scan, if_neg (by decide), innerCandidates94_12_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_13
    (if 3 ≤ innerCandidates94_12.length then search edge 3 [509, 3359] innerCandidates94_12 3868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_13

def innerTail94_11 : List Nat := 2111 :: innerTail94_12
def innerCandidates94_11 : List Nat := [7793]
set_option maxRecDepth 4096 in
theorem innerCandidates94_11_checked : candidates edge 4 509 2111 innerTail94_12 best.bound = innerCandidates94_11 := by
  unfold candidates
  have row : edge 2111 = PrimePairCache.smallRow2111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_11 : scan edge (search edge 3) 4 [509] 509 innerTail94_11 best = best := by
  rw [innerTail94_11, scan, if_neg (by decide), innerCandidates94_11_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_12
    (if 3 ≤ innerCandidates94_11.length then search edge 3 [509, 2111] innerCandidates94_11 2620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_12

def innerTail94_10 : List Nat := 1973 :: innerTail94_11
def innerCandidates94_10 : List Nat := [5711]
set_option maxRecDepth 4096 in
theorem innerCandidates94_10_checked : candidates edge 4 509 1973 innerTail94_11 best.bound = innerCandidates94_10 := by
  unfold candidates
  have row : edge 1973 = PrimePairCache.smallRow1973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_10 : scan edge (search edge 3) 4 [509] 509 innerTail94_10 best = best := by
  rw [innerTail94_10, scan, if_neg (by decide), innerCandidates94_10_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_11
    (if 3 ≤ innerCandidates94_10.length then search edge 3 [509, 1973] innerCandidates94_10 2482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_11

def innerTail94_9 : List Nat := 1613 :: innerTail94_10
def innerCandidates94_9 : List Nat := [4673]
set_option maxRecDepth 4096 in
theorem innerCandidates94_9_checked : candidates edge 4 509 1613 innerTail94_10 best.bound = innerCandidates94_9 := by
  unfold candidates
  have row : edge 1613 = PrimePairCache.smallRow1613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_9 : scan edge (search edge 3) 4 [509] 509 innerTail94_9 best = best := by
  rw [innerTail94_9, scan, if_neg (by decide), innerCandidates94_9_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_10
    (if 3 ≤ innerCandidates94_9.length then search edge 3 [509, 1613] innerCandidates94_9 2122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_10

def innerTail94_8 : List Nat := 1553 :: innerTail94_9
def innerCandidates94_8 : List Nat := [1613]
set_option maxRecDepth 4096 in
theorem innerCandidates94_8_checked : candidates edge 4 509 1553 innerTail94_9 best.bound = innerCandidates94_8 := by
  unfold candidates
  have row : edge 1553 = PrimePairCache.smallRow1553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_8 : scan edge (search edge 3) 4 [509] 509 innerTail94_8 best = best := by
  rw [innerTail94_8, scan, if_neg (by decide), innerCandidates94_8_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_9
    (if 3 ≤ innerCandidates94_8.length then search edge 3 [509, 1553] innerCandidates94_8 2062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_9

def innerTail94_7 : List Nat := 1451 :: innerTail94_8
def innerCandidates94_7 : List Nat := [7193]
set_option maxRecDepth 4096 in
theorem innerCandidates94_7_checked : candidates edge 4 509 1451 innerTail94_8 best.bound = innerCandidates94_7 := by
  unfold candidates
  have row : edge 1451 = PrimePairCache.smallRow1451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_7 : scan edge (search edge 3) 4 [509] 509 innerTail94_7 best = best := by
  rw [innerTail94_7, scan, if_neg (by decide), innerCandidates94_7_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_8
    (if 3 ≤ innerCandidates94_7.length then search edge 3 [509, 1451] innerCandidates94_7 1960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_8

def innerTail94_6 : List Nat := 1091 :: innerTail94_7
def innerCandidates94_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates94_6_checked : candidates edge 4 509 1091 innerTail94_7 best.bound = innerCandidates94_6 := by
  unfold candidates
  have row : edge 1091 = PrimePairCache.smallRow1091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_6 : scan edge (search edge 3) 4 [509] 509 innerTail94_6 best = best := by
  rw [innerTail94_6, scan, if_neg (by decide), innerCandidates94_6_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_7
    (if 3 ≤ innerCandidates94_6.length then search edge 3 [509, 1091] innerCandidates94_6 1600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_7

def innerTail94_5 : List Nat := 1061 :: innerTail94_6
def innerCandidates94_5 : List Nat := [8117]
set_option maxRecDepth 4096 in
theorem innerCandidates94_5_checked : candidates edge 4 509 1061 innerTail94_6 best.bound = innerCandidates94_5 := by
  unfold candidates
  have row : edge 1061 = PrimePairCache.smallRow1061.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_5 : scan edge (search edge 3) 4 [509] 509 innerTail94_5 best = best := by
  rw [innerTail94_5, scan, if_neg (by decide), innerCandidates94_5_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_6
    (if 3 ≤ innerCandidates94_5.length then search edge 3 [509, 1061] innerCandidates94_5 1570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_6

def innerTail94_4 : List Nat := 1019 :: innerTail94_5
def innerCandidates94_4 : List Nat := [3929, 5483, 7193]
set_option maxRecDepth 4096 in
theorem innerCandidates94_4_checked : candidates edge 4 509 1019 innerTail94_5 best.bound = innerCandidates94_4 := by
  unfold candidates
  have row : edge 1019 = PrimePairCache.smallRow1019.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild94_4 : search edge 3 [509, 1019] innerCandidates94_4 1528 best = best := by decide

theorem innerScan94_4 : scan edge (search edge 3) 4 [509] 509 innerTail94_4 best = best := by
  rw [innerTail94_4, scan, if_neg (by decide), innerCandidates94_4_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_5
    (if 3 ≤ innerCandidates94_4.length then search edge 3 [509, 1019] innerCandidates94_4 1528 best else best) = best
  rw [if_pos (by decide), innerChild94_4]
  exact innerScan94_5

def innerTail94_3 : List Nat := 947 :: innerTail94_4
def innerCandidates94_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates94_3_checked : candidates edge 4 509 947 innerTail94_4 best.bound = innerCandidates94_3 := by
  unfold candidates
  have row : edge 947 = PrimePairCache.smallRow947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_3 : scan edge (search edge 3) 4 [509] 509 innerTail94_3 best = best := by
  rw [innerTail94_3, scan, if_neg (by decide), innerCandidates94_3_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_4
    (if 3 ≤ innerCandidates94_3.length then search edge 3 [509, 947] innerCandidates94_3 1456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_4

def innerTail94_2 : List Nat := 863 :: innerTail94_3
def innerCandidates94_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates94_2_checked : candidates edge 4 509 863 innerTail94_3 best.bound = innerCandidates94_2 := by
  unfold candidates
  have row : edge 863 = PrimePairCache.smallRow863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_2 : scan edge (search edge 3) 4 [509] 509 innerTail94_2 best = best := by
  rw [innerTail94_2, scan, if_neg (by decide), innerCandidates94_2_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_3
    (if 3 ≤ innerCandidates94_2.length then search edge 3 [509, 863] innerCandidates94_2 1372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_3

def innerTail94_1 : List Nat := 797 :: innerTail94_2
def innerCandidates94_1 : List Nat := [3929]
set_option maxRecDepth 4096 in
theorem innerCandidates94_1_checked : candidates edge 4 509 797 innerTail94_2 best.bound = innerCandidates94_1 := by
  unfold candidates
  have row : edge 797 = PrimePairCache.smallRow797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan94_1 : scan edge (search edge 3) 4 [509] 509 innerTail94_1 best = best := by
  rw [innerTail94_1, scan, if_neg (by decide), innerCandidates94_1_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_2
    (if 3 ≤ innerCandidates94_1.length then search edge 3 [509, 797] innerCandidates94_1 1306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan94_2

def innerTail94_0 : List Nat := 647 :: innerTail94_1
def innerCandidates94_0 : List Nat := [1019, 4241, 5471]
set_option maxRecDepth 4096 in
theorem innerCandidates94_0_checked : candidates edge 4 509 647 innerTail94_1 best.bound = innerCandidates94_0 := by
  unfold candidates
  have row : edge 647 = PrimePairCache.smallRow647.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild94_0 : search edge 3 [509, 647] innerCandidates94_0 1156 best = best := by decide

theorem innerScan94_0 : scan edge (search edge 3) 4 [509] 509 innerTail94_0 best = best := by
  rw [innerTail94_0, scan, if_neg (by decide), innerCandidates94_0_checked]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_1
    (if 3 ≤ innerCandidates94_0.length then search edge 3 [509, 647] innerCandidates94_0 1156 best else best) = best
  rw [if_pos (by decide), innerChild94_0]
  exact innerScan94_1

theorem rootBranch94 : search edge 4 [509] rootCandidates94 509 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [509] 509 innerTail94_0 best = best
  exact innerScan94_0

def rootCandidates93 : List Nat := [563, 647, 653, 911, 1193, 1601, 1997, 2411, 2711, 3299, 3593, 4229, 5003, 5021, 5309, 5351, 5507, 5843, 5903, 7673, 7817]
set_option maxRecDepth 4096 in
theorem rootCandidates93_checked : candidates edge 5 0 503 tail94 best.bound = rootCandidates93 := by
  unfold candidates
  have row : edge 503 = PrimePairCache.smallRow503.contains := by funext q; rfl
  rw [row]
  decide

def innerTail93_21 : List Nat := []
theorem innerScan93_21 : scan edge (search edge 3) 4 [503] 503 innerTail93_21 best = best := rfl

def innerTail93_20 : List Nat := 7817 :: innerTail93_21
def innerCandidates93_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates93_20_checked : candidates edge 4 503 7817 innerTail93_21 best.bound = innerCandidates93_20 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_20 : scan edge (search edge 3) 4 [503] 503 innerTail93_20 best = best := by
  rw [innerTail93_20, scan, if_neg (by decide), innerCandidates93_20_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_21
    (if 3 ≤ innerCandidates93_20.length then search edge 3 [503, 7817] innerCandidates93_20 8320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_21

def innerTail93_19 : List Nat := 7673 :: innerTail93_20
def innerCandidates93_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates93_19_checked : candidates edge 4 503 7673 innerTail93_20 best.bound = innerCandidates93_19 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_19 : scan edge (search edge 3) 4 [503] 503 innerTail93_19 best = best := by
  rw [innerTail93_19, scan, if_neg (by decide), innerCandidates93_19_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_20
    (if 3 ≤ innerCandidates93_19.length then search edge 3 [503, 7673] innerCandidates93_19 8176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_20

def innerTail93_18 : List Nat := 5903 :: innerTail93_19
def innerCandidates93_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates93_18_checked : candidates edge 4 503 5903 innerTail93_19 best.bound = innerCandidates93_18 := by
  unfold candidates
  have row : edge 5903 = SearchData.row5903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_18 : scan edge (search edge 3) 4 [503] 503 innerTail93_18 best = best := by
  rw [innerTail93_18, scan, if_neg (by decide), innerCandidates93_18_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_19
    (if 3 ≤ innerCandidates93_18.length then search edge 3 [503, 5903] innerCandidates93_18 6406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_19

def innerTail93_17 : List Nat := 5843 :: innerTail93_18
def innerCandidates93_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates93_17_checked : candidates edge 4 503 5843 innerTail93_18 best.bound = innerCandidates93_17 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_17 : scan edge (search edge 3) 4 [503] 503 innerTail93_17 best = best := by
  rw [innerTail93_17, scan, if_neg (by decide), innerCandidates93_17_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_18
    (if 3 ≤ innerCandidates93_17.length then search edge 3 [503, 5843] innerCandidates93_17 6346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_18

def innerTail93_16 : List Nat := 5507 :: innerTail93_17
def innerCandidates93_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates93_16_checked : candidates edge 4 503 5507 innerTail93_17 best.bound = innerCandidates93_16 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_16 : scan edge (search edge 3) 4 [503] 503 innerTail93_16 best = best := by
  rw [innerTail93_16, scan, if_neg (by decide), innerCandidates93_16_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_17
    (if 3 ≤ innerCandidates93_16.length then search edge 3 [503, 5507] innerCandidates93_16 6010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_17

def innerTail93_15 : List Nat := 5351 :: innerTail93_16
def innerCandidates93_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates93_15_checked : candidates edge 4 503 5351 innerTail93_16 best.bound = innerCandidates93_15 := by
  unfold candidates
  have row : edge 5351 = SearchData.row5351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_15 : scan edge (search edge 3) 4 [503] 503 innerTail93_15 best = best := by
  rw [innerTail93_15, scan, if_neg (by decide), innerCandidates93_15_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_16
    (if 3 ≤ innerCandidates93_15.length then search edge 3 [503, 5351] innerCandidates93_15 5854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_16

def innerTail93_14 : List Nat := 5309 :: innerTail93_15
def innerCandidates93_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates93_14_checked : candidates edge 4 503 5309 innerTail93_15 best.bound = innerCandidates93_14 := by
  unfold candidates
  have row : edge 5309 = SearchData.row5309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_14 : scan edge (search edge 3) 4 [503] 503 innerTail93_14 best = best := by
  rw [innerTail93_14, scan, if_neg (by decide), innerCandidates93_14_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_15
    (if 3 ≤ innerCandidates93_14.length then search edge 3 [503, 5309] innerCandidates93_14 5812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_15

def innerTail93_13 : List Nat := 5021 :: innerTail93_14
def innerCandidates93_13 : List Nat := [5507]
set_option maxRecDepth 4096 in
theorem innerCandidates93_13_checked : candidates edge 4 503 5021 innerTail93_14 best.bound = innerCandidates93_13 := by
  unfold candidates
  have row : edge 5021 = SearchData.row5021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_13 : scan edge (search edge 3) 4 [503] 503 innerTail93_13 best = best := by
  rw [innerTail93_13, scan, if_neg (by decide), innerCandidates93_13_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_14
    (if 3 ≤ innerCandidates93_13.length then search edge 3 [503, 5021] innerCandidates93_13 5524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_14

def innerTail93_12 : List Nat := 5003 :: innerTail93_13
def innerCandidates93_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates93_12_checked : candidates edge 4 503 5003 innerTail93_13 best.bound = innerCandidates93_12 := by
  unfold candidates
  have row : edge 5003 = SearchData.row5003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_12 : scan edge (search edge 3) 4 [503] 503 innerTail93_12 best = best := by
  rw [innerTail93_12, scan, if_neg (by decide), innerCandidates93_12_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_13
    (if 3 ≤ innerCandidates93_12.length then search edge 3 [503, 5003] innerCandidates93_12 5506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_13

def innerTail93_11 : List Nat := 4229 :: innerTail93_12
def innerCandidates93_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates93_11_checked : candidates edge 4 503 4229 innerTail93_12 best.bound = innerCandidates93_11 := by
  unfold candidates
  have row : edge 4229 = SearchData.row4229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_11 : scan edge (search edge 3) 4 [503] 503 innerTail93_11 best = best := by
  rw [innerTail93_11, scan, if_neg (by decide), innerCandidates93_11_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_12
    (if 3 ≤ innerCandidates93_11.length then search edge 3 [503, 4229] innerCandidates93_11 4732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_12

def innerTail93_10 : List Nat := 3593 :: innerTail93_11
def innerCandidates93_10 : List Nat := [5021, 5903]
set_option maxRecDepth 4096 in
theorem innerCandidates93_10_checked : candidates edge 4 503 3593 innerTail93_11 best.bound = innerCandidates93_10 := by
  unfold candidates
  have row : edge 3593 = SearchData.row3593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_10 : scan edge (search edge 3) 4 [503] 503 innerTail93_10 best = best := by
  rw [innerTail93_10, scan, if_neg (by decide), innerCandidates93_10_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_11
    (if 3 ≤ innerCandidates93_10.length then search edge 3 [503, 3593] innerCandidates93_10 4096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_11

def innerTail93_9 : List Nat := 3299 :: innerTail93_10
def innerCandidates93_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates93_9_checked : candidates edge 4 503 3299 innerTail93_10 best.bound = innerCandidates93_9 := by
  unfold candidates
  have row : edge 3299 = SearchData.row3299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_9 : scan edge (search edge 3) 4 [503] 503 innerTail93_9 best = best := by
  rw [innerTail93_9, scan, if_neg (by decide), innerCandidates93_9_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_10
    (if 3 ≤ innerCandidates93_9.length then search edge 3 [503, 3299] innerCandidates93_9 3802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_10

def innerTail93_8 : List Nat := 2711 :: innerTail93_9
def innerCandidates93_8 : List Nat := [7817]
set_option maxRecDepth 4096 in
theorem innerCandidates93_8_checked : candidates edge 4 503 2711 innerTail93_9 best.bound = innerCandidates93_8 := by
  unfold candidates
  have row : edge 2711 = SearchData.row2711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_8 : scan edge (search edge 3) 4 [503] 503 innerTail93_8 best = best := by
  rw [innerTail93_8, scan, if_neg (by decide), innerCandidates93_8_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_9
    (if 3 ≤ innerCandidates93_8.length then search edge 3 [503, 2711] innerCandidates93_8 3214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_9

def innerTail93_7 : List Nat := 2411 :: innerTail93_8
def innerCandidates93_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates93_7_checked : candidates edge 4 503 2411 innerTail93_8 best.bound = innerCandidates93_7 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_7 : scan edge (search edge 3) 4 [503] 503 innerTail93_7 best = best := by
  rw [innerTail93_7, scan, if_neg (by decide), innerCandidates93_7_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_8
    (if 3 ≤ innerCandidates93_7.length then search edge 3 [503, 2411] innerCandidates93_7 2914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_8

def innerTail93_6 : List Nat := 1997 :: innerTail93_7
def innerCandidates93_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates93_6_checked : candidates edge 4 503 1997 innerTail93_7 best.bound = innerCandidates93_6 := by
  unfold candidates
  have row : edge 1997 = PrimePairCache.smallRow1997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_6 : scan edge (search edge 3) 4 [503] 503 innerTail93_6 best = best := by
  rw [innerTail93_6, scan, if_neg (by decide), innerCandidates93_6_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_7
    (if 3 ≤ innerCandidates93_6.length then search edge 3 [503, 1997] innerCandidates93_6 2500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_7

def innerTail93_5 : List Nat := 1601 :: innerTail93_6
def innerCandidates93_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates93_5_checked : candidates edge 4 503 1601 innerTail93_6 best.bound = innerCandidates93_5 := by
  unfold candidates
  have row : edge 1601 = PrimePairCache.smallRow1601.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_5 : scan edge (search edge 3) 4 [503] 503 innerTail93_5 best = best := by
  rw [innerTail93_5, scan, if_neg (by decide), innerCandidates93_5_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_6
    (if 3 ≤ innerCandidates93_5.length then search edge 3 [503, 1601] innerCandidates93_5 2104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_6

def innerTail93_4 : List Nat := 1193 :: innerTail93_5
def innerCandidates93_4 : List Nat := [5507]
set_option maxRecDepth 4096 in
theorem innerCandidates93_4_checked : candidates edge 4 503 1193 innerTail93_5 best.bound = innerCandidates93_4 := by
  unfold candidates
  have row : edge 1193 = PrimePairCache.smallRow1193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_4 : scan edge (search edge 3) 4 [503] 503 innerTail93_4 best = best := by
  rw [innerTail93_4, scan, if_neg (by decide), innerCandidates93_4_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_5
    (if 3 ≤ innerCandidates93_4.length then search edge 3 [503, 1193] innerCandidates93_4 1696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_5

def innerTail93_3 : List Nat := 911 :: innerTail93_4
def innerCandidates93_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates93_3_checked : candidates edge 4 503 911 innerTail93_4 best.bound = innerCandidates93_3 := by
  unfold candidates
  have row : edge 911 = PrimePairCache.smallRow911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_3 : scan edge (search edge 3) 4 [503] 503 innerTail93_3 best = best := by
  rw [innerTail93_3, scan, if_neg (by decide), innerCandidates93_3_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_4
    (if 3 ≤ innerCandidates93_3.length then search edge 3 [503, 911] innerCandidates93_3 1414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_4

def innerTail93_2 : List Nat := 653 :: innerTail93_3
def innerCandidates93_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates93_2_checked : candidates edge 4 503 653 innerTail93_3 best.bound = innerCandidates93_2 := by
  unfold candidates
  have row : edge 653 = PrimePairCache.smallRow653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_2 : scan edge (search edge 3) 4 [503] 503 innerTail93_2 best = best := by
  rw [innerTail93_2, scan, if_neg (by decide), innerCandidates93_2_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_3
    (if 3 ≤ innerCandidates93_2.length then search edge 3 [503, 653] innerCandidates93_2 1156 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_3

def innerTail93_1 : List Nat := 647 :: innerTail93_2
def innerCandidates93_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates93_1_checked : candidates edge 4 503 647 innerTail93_2 best.bound = innerCandidates93_1 := by
  unfold candidates
  have row : edge 647 = PrimePairCache.smallRow647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_1 : scan edge (search edge 3) 4 [503] 503 innerTail93_1 best = best := by
  rw [innerTail93_1, scan, if_neg (by decide), innerCandidates93_1_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_2
    (if 3 ≤ innerCandidates93_1.length then search edge 3 [503, 647] innerCandidates93_1 1150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_2

def innerTail93_0 : List Nat := 563 :: innerTail93_1
def innerCandidates93_0 : List Nat := [1601]
set_option maxRecDepth 4096 in
theorem innerCandidates93_0_checked : candidates edge 4 503 563 innerTail93_1 best.bound = innerCandidates93_0 := by
  unfold candidates
  have row : edge 563 = PrimePairCache.smallRow563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan93_0 : scan edge (search edge 3) 4 [503] 503 innerTail93_0 best = best := by
  rw [innerTail93_0, scan, if_neg (by decide), innerCandidates93_0_checked]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_1
    (if 3 ≤ innerCandidates93_0.length then search edge 3 [503, 563] innerCandidates93_0 1066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan93_1

theorem rootBranch93 : search edge 4 [503] rootCandidates93 503 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [503] 503 innerTail93_0 best = best
  exact innerScan93_0

end ProjectEulerSolutions.PrimePairUpper.Search
