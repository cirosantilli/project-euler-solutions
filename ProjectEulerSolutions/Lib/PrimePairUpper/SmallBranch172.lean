import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates167 : List Nat := [1091, 1097, 1697, 1721, 1811, 2273, 2543, 3659, 3803, 4133, 4139, 4259, 4349, 5039, 5087, 6047, 6221, 6323, 6719, 6761, 7187, 7247, 7331, 7433, 7577, 7937, 8231]
set_option maxRecDepth 4096 in
theorem rootCandidates167_checked : candidates edge 5 0 1013 tail168 best.bound = rootCandidates167 := by
  unfold candidates
  have row : edge 1013 = PrimePairCache.smallRow1013.contains := by funext q; rfl
  rw [row]
  decide

def innerTail167_27 : List Nat := []
theorem innerScan167_27 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_27 best = best := rfl

def innerTail167_26 : List Nat := 8231 :: innerTail167_27
def innerCandidates167_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates167_26_checked : candidates edge 4 1013 8231 innerTail167_27 best.bound = innerCandidates167_26 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_26 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_26 best = best := by
  rw [innerTail167_26, scan, if_neg (by decide), innerCandidates167_26_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_27
    (if 3 ≤ innerCandidates167_26.length then search edge 3 [1013, 8231] innerCandidates167_26 9244 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_27

def innerTail167_25 : List Nat := 7937 :: innerTail167_26
def innerCandidates167_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates167_25_checked : candidates edge 4 1013 7937 innerTail167_26 best.bound = innerCandidates167_25 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_25 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_25 best = best := by
  rw [innerTail167_25, scan, if_neg (by decide), innerCandidates167_25_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_26
    (if 3 ≤ innerCandidates167_25.length then search edge 3 [1013, 7937] innerCandidates167_25 8950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_26

def innerTail167_24 : List Nat := 7577 :: innerTail167_25
def innerCandidates167_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates167_24_checked : candidates edge 4 1013 7577 innerTail167_25 best.bound = innerCandidates167_24 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_24 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_24 best = best := by
  rw [innerTail167_24, scan, if_neg (by decide), innerCandidates167_24_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_25
    (if 3 ≤ innerCandidates167_24.length then search edge 3 [1013, 7577] innerCandidates167_24 8590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_25

def innerTail167_23 : List Nat := 7433 :: innerTail167_24
def innerCandidates167_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates167_23_checked : candidates edge 4 1013 7433 innerTail167_24 best.bound = innerCandidates167_23 := by
  unfold candidates
  have row : edge 7433 = SearchData.row7433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_23 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_23 best = best := by
  rw [innerTail167_23, scan, if_neg (by decide), innerCandidates167_23_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_24
    (if 3 ≤ innerCandidates167_23.length then search edge 3 [1013, 7433] innerCandidates167_23 8446 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_24

def innerTail167_22 : List Nat := 7331 :: innerTail167_23
def innerCandidates167_22 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates167_22_checked : candidates edge 4 1013 7331 innerTail167_23 best.bound = innerCandidates167_22 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_22 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_22 best = best := by
  rw [innerTail167_22, scan, if_neg (by decide), innerCandidates167_22_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_23
    (if 3 ≤ innerCandidates167_22.length then search edge 3 [1013, 7331] innerCandidates167_22 8344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_23

def innerTail167_21 : List Nat := 7247 :: innerTail167_22
def innerCandidates167_21 : List Nat := [7331]
set_option maxRecDepth 4096 in
theorem innerCandidates167_21_checked : candidates edge 4 1013 7247 innerTail167_22 best.bound = innerCandidates167_21 := by
  unfold candidates
  have row : edge 7247 = SearchData.row7247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_21 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_21 best = best := by
  rw [innerTail167_21, scan, if_neg (by decide), innerCandidates167_21_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_22
    (if 3 ≤ innerCandidates167_21.length then search edge 3 [1013, 7247] innerCandidates167_21 8260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_22

def innerTail167_20 : List Nat := 7187 :: innerTail167_21
def innerCandidates167_20 : List Nat := [7331]
set_option maxRecDepth 4096 in
theorem innerCandidates167_20_checked : candidates edge 4 1013 7187 innerTail167_21 best.bound = innerCandidates167_20 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_20 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_20 best = best := by
  rw [innerTail167_20, scan, if_neg (by decide), innerCandidates167_20_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_21
    (if 3 ≤ innerCandidates167_20.length then search edge 3 [1013, 7187] innerCandidates167_20 8200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_21

def innerTail167_19 : List Nat := 6761 :: innerTail167_20
def innerCandidates167_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates167_19_checked : candidates edge 4 1013 6761 innerTail167_20 best.bound = innerCandidates167_19 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_19 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_19 best = best := by
  rw [innerTail167_19, scan, if_neg (by decide), innerCandidates167_19_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_20
    (if 3 ≤ innerCandidates167_19.length then search edge 3 [1013, 6761] innerCandidates167_19 7774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_20

def innerTail167_18 : List Nat := 6719 :: innerTail167_19
def innerCandidates167_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates167_18_checked : candidates edge 4 1013 6719 innerTail167_19 best.bound = innerCandidates167_18 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_18 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_18 best = best := by
  rw [innerTail167_18, scan, if_neg (by decide), innerCandidates167_18_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_19
    (if 3 ≤ innerCandidates167_18.length then search edge 3 [1013, 6719] innerCandidates167_18 7732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_19

def innerTail167_17 : List Nat := 6323 :: innerTail167_18
def innerCandidates167_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates167_17_checked : candidates edge 4 1013 6323 innerTail167_18 best.bound = innerCandidates167_17 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_17 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_17 best = best := by
  rw [innerTail167_17, scan, if_neg (by decide), innerCandidates167_17_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_18
    (if 3 ≤ innerCandidates167_17.length then search edge 3 [1013, 6323] innerCandidates167_17 7336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_18

def innerTail167_16 : List Nat := 6221 :: innerTail167_17
def innerCandidates167_16 : List Nat := [7577]
set_option maxRecDepth 4096 in
theorem innerCandidates167_16_checked : candidates edge 4 1013 6221 innerTail167_17 best.bound = innerCandidates167_16 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_16 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_16 best = best := by
  rw [innerTail167_16, scan, if_neg (by decide), innerCandidates167_16_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_17
    (if 3 ≤ innerCandidates167_16.length then search edge 3 [1013, 6221] innerCandidates167_16 7234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_17

def innerTail167_15 : List Nat := 6047 :: innerTail167_16
def innerCandidates167_15 : List Nat := [6719]
set_option maxRecDepth 4096 in
theorem innerCandidates167_15_checked : candidates edge 4 1013 6047 innerTail167_16 best.bound = innerCandidates167_15 := by
  unfold candidates
  have row : edge 6047 = SearchData.row6047.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_15 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_15 best = best := by
  rw [innerTail167_15, scan, if_neg (by decide), innerCandidates167_15_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_16
    (if 3 ≤ innerCandidates167_15.length then search edge 3 [1013, 6047] innerCandidates167_15 7060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_16

def innerTail167_14 : List Nat := 5087 :: innerTail167_15
def innerCandidates167_14 : List Nat := [6719, 7247]
set_option maxRecDepth 4096 in
theorem innerCandidates167_14_checked : candidates edge 4 1013 5087 innerTail167_15 best.bound = innerCandidates167_14 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_14 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_14 best = best := by
  rw [innerTail167_14, scan, if_neg (by decide), innerCandidates167_14_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_15
    (if 3 ≤ innerCandidates167_14.length then search edge 3 [1013, 5087] innerCandidates167_14 6100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_15

def innerTail167_13 : List Nat := 5039 :: innerTail167_14
def innerCandidates167_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates167_13_checked : candidates edge 4 1013 5039 innerTail167_14 best.bound = innerCandidates167_13 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_13 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_13 best = best := by
  rw [innerTail167_13, scan, if_neg (by decide), innerCandidates167_13_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_14
    (if 3 ≤ innerCandidates167_13.length then search edge 3 [1013, 5039] innerCandidates167_13 6052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_14

def innerTail167_12 : List Nat := 4349 :: innerTail167_13
def innerCandidates167_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates167_12_checked : candidates edge 4 1013 4349 innerTail167_13 best.bound = innerCandidates167_12 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_12 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_12 best = best := by
  rw [innerTail167_12, scan, if_neg (by decide), innerCandidates167_12_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_13
    (if 3 ≤ innerCandidates167_12.length then search edge 3 [1013, 4349] innerCandidates167_12 5362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_13

def innerTail167_11 : List Nat := 4259 :: innerTail167_12
def innerCandidates167_11 : List Nat := [5087, 6221]
set_option maxRecDepth 4096 in
theorem innerCandidates167_11_checked : candidates edge 4 1013 4259 innerTail167_12 best.bound = innerCandidates167_11 := by
  unfold candidates
  have row : edge 4259 = SearchData.row4259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_11 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_11 best = best := by
  rw [innerTail167_11, scan, if_neg (by decide), innerCandidates167_11_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_12
    (if 3 ≤ innerCandidates167_11.length then search edge 3 [1013, 4259] innerCandidates167_11 5272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_12

def innerTail167_10 : List Nat := 4139 :: innerTail167_11
def innerCandidates167_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates167_10_checked : candidates edge 4 1013 4139 innerTail167_11 best.bound = innerCandidates167_10 := by
  unfold candidates
  have row : edge 4139 = SearchData.row4139.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_10 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_10 best = best := by
  rw [innerTail167_10, scan, if_neg (by decide), innerCandidates167_10_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_11
    (if 3 ≤ innerCandidates167_10.length then search edge 3 [1013, 4139] innerCandidates167_10 5152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_11

def innerTail167_9 : List Nat := 4133 :: innerTail167_10
def innerCandidates167_9 : List Nat := [7433]
set_option maxRecDepth 4096 in
theorem innerCandidates167_9_checked : candidates edge 4 1013 4133 innerTail167_10 best.bound = innerCandidates167_9 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_9 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_9 best = best := by
  rw [innerTail167_9, scan, if_neg (by decide), innerCandidates167_9_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_10
    (if 3 ≤ innerCandidates167_9.length then search edge 3 [1013, 4133] innerCandidates167_9 5146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_10

def innerTail167_8 : List Nat := 3803 :: innerTail167_9
def innerCandidates167_8 : List Nat := [5087]
set_option maxRecDepth 4096 in
theorem innerCandidates167_8_checked : candidates edge 4 1013 3803 innerTail167_9 best.bound = innerCandidates167_8 := by
  unfold candidates
  have row : edge 3803 = SearchData.row3803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_8 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_8 best = best := by
  rw [innerTail167_8, scan, if_neg (by decide), innerCandidates167_8_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_9
    (if 3 ≤ innerCandidates167_8.length then search edge 3 [1013, 3803] innerCandidates167_8 4816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_9

def innerTail167_7 : List Nat := 3659 :: innerTail167_8
def innerCandidates167_7 : List Nat := [3803, 6047]
set_option maxRecDepth 4096 in
theorem innerCandidates167_7_checked : candidates edge 4 1013 3659 innerTail167_8 best.bound = innerCandidates167_7 := by
  unfold candidates
  have row : edge 3659 = SearchData.row3659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_7 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_7 best = best := by
  rw [innerTail167_7, scan, if_neg (by decide), innerCandidates167_7_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_8
    (if 3 ≤ innerCandidates167_7.length then search edge 3 [1013, 3659] innerCandidates167_7 4672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_8

def innerTail167_6 : List Nat := 2543 :: innerTail167_7
def innerCandidates167_6 : List Nat := [4349, 7331]
set_option maxRecDepth 4096 in
theorem innerCandidates167_6_checked : candidates edge 4 1013 2543 innerTail167_7 best.bound = innerCandidates167_6 := by
  unfold candidates
  have row : edge 2543 = PrimePairCache.smallRow2543.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_6 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_6 best = best := by
  rw [innerTail167_6, scan, if_neg (by decide), innerCandidates167_6_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_7
    (if 3 ≤ innerCandidates167_6.length then search edge 3 [1013, 2543] innerCandidates167_6 3556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_7

def innerTail167_5 : List Nat := 2273 :: innerTail167_6
def innerCandidates167_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates167_5_checked : candidates edge 4 1013 2273 innerTail167_6 best.bound = innerCandidates167_5 := by
  unfold candidates
  have row : edge 2273 = PrimePairCache.smallRow2273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_5 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_5 best = best := by
  rw [innerTail167_5, scan, if_neg (by decide), innerCandidates167_5_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_6
    (if 3 ≤ innerCandidates167_5.length then search edge 3 [1013, 2273] innerCandidates167_5 3286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_6

def innerTail167_4 : List Nat := 1811 :: innerTail167_5
def innerCandidates167_4 : List Nat := [5039]
set_option maxRecDepth 4096 in
theorem innerCandidates167_4_checked : candidates edge 4 1013 1811 innerTail167_5 best.bound = innerCandidates167_4 := by
  unfold candidates
  have row : edge 1811 = PrimePairCache.smallRow1811.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_4 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_4 best = best := by
  rw [innerTail167_4, scan, if_neg (by decide), innerCandidates167_4_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_5
    (if 3 ≤ innerCandidates167_4.length then search edge 3 [1013, 1811] innerCandidates167_4 2824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_5

def innerTail167_3 : List Nat := 1721 :: innerTail167_4
def innerCandidates167_3 : List Nat := [8231]
set_option maxRecDepth 4096 in
theorem innerCandidates167_3_checked : candidates edge 4 1013 1721 innerTail167_4 best.bound = innerCandidates167_3 := by
  unfold candidates
  have row : edge 1721 = PrimePairCache.smallRow1721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_3 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_3 best = best := by
  rw [innerTail167_3, scan, if_neg (by decide), innerCandidates167_3_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_4
    (if 3 ≤ innerCandidates167_3.length then search edge 3 [1013, 1721] innerCandidates167_3 2734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_4

def innerTail167_2 : List Nat := 1697 :: innerTail167_3
def innerCandidates167_2 : List Nat := [5087]
set_option maxRecDepth 4096 in
theorem innerCandidates167_2_checked : candidates edge 4 1013 1697 innerTail167_3 best.bound = innerCandidates167_2 := by
  unfold candidates
  have row : edge 1697 = PrimePairCache.smallRow1697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_2 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_2 best = best := by
  rw [innerTail167_2, scan, if_neg (by decide), innerCandidates167_2_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_3
    (if 3 ≤ innerCandidates167_2.length then search edge 3 [1013, 1697] innerCandidates167_2 2710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_3

def innerTail167_1 : List Nat := 1097 :: innerTail167_2
def innerCandidates167_1 : List Nat := [5087, 6323]
set_option maxRecDepth 4096 in
theorem innerCandidates167_1_checked : candidates edge 4 1013 1097 innerTail167_2 best.bound = innerCandidates167_1 := by
  unfold candidates
  have row : edge 1097 = PrimePairCache.smallRow1097.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_1 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_1 best = best := by
  rw [innerTail167_1, scan, if_neg (by decide), innerCandidates167_1_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_2
    (if 3 ≤ innerCandidates167_1.length then search edge 3 [1013, 1097] innerCandidates167_1 2110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_2

def innerTail167_0 : List Nat := 1091 :: innerTail167_1
def innerCandidates167_0 : List Nat := [1697]
set_option maxRecDepth 4096 in
theorem innerCandidates167_0_checked : candidates edge 4 1013 1091 innerTail167_1 best.bound = innerCandidates167_0 := by
  unfold candidates
  have row : edge 1091 = PrimePairCache.smallRow1091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan167_0 : scan edge (search edge 3) 4 [1013] 1013 innerTail167_0 best = best := by
  rw [innerTail167_0, scan, if_neg (by decide), innerCandidates167_0_checked]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_1
    (if 3 ≤ innerCandidates167_0.length then search edge 3 [1013, 1091] innerCandidates167_0 2104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan167_1

theorem rootBranch167 : search edge 4 [1013] rootCandidates167 1013 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1013] 1013 innerTail167_0 best = best
  exact innerScan167_0

def rootCandidates166 : List Nat := [1789, 1933, 2221, 2473, 2851, 3613, 3823, 4099, 4657, 5653, 6043, 6367, 6397, 6481, 6577, 6673, 7027, 7309, 7393, 7753]
set_option maxRecDepth 4096 in
theorem rootCandidates166_checked : candidates edge 5 0 1009 tail167 best.bound = rootCandidates166 := by
  unfold candidates
  have row : edge 1009 = PrimePairCache.smallRow1009.contains := by funext q; rfl
  rw [row]
  decide

def innerTail166_20 : List Nat := []
theorem innerScan166_20 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_20 best = best := rfl

def innerTail166_19 : List Nat := 7753 :: innerTail166_20
def innerCandidates166_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates166_19_checked : candidates edge 4 1009 7753 innerTail166_20 best.bound = innerCandidates166_19 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_19 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_19 best = best := by
  rw [innerTail166_19, scan, if_neg (by decide), innerCandidates166_19_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_20
    (if 3 ≤ innerCandidates166_19.length then search edge 3 [1009, 7753] innerCandidates166_19 8762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_20

def innerTail166_18 : List Nat := 7393 :: innerTail166_19
def innerCandidates166_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates166_18_checked : candidates edge 4 1009 7393 innerTail166_19 best.bound = innerCandidates166_18 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_18 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_18 best = best := by
  rw [innerTail166_18, scan, if_neg (by decide), innerCandidates166_18_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_19
    (if 3 ≤ innerCandidates166_18.length then search edge 3 [1009, 7393] innerCandidates166_18 8402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_19

def innerTail166_17 : List Nat := 7309 :: innerTail166_18
def innerCandidates166_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates166_17_checked : candidates edge 4 1009 7309 innerTail166_18 best.bound = innerCandidates166_17 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_17 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_17 best = best := by
  rw [innerTail166_17, scan, if_neg (by decide), innerCandidates166_17_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_18
    (if 3 ≤ innerCandidates166_17.length then search edge 3 [1009, 7309] innerCandidates166_17 8318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_18

def innerTail166_16 : List Nat := 7027 :: innerTail166_17
def innerCandidates166_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates166_16_checked : candidates edge 4 1009 7027 innerTail166_17 best.bound = innerCandidates166_16 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_16 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_16 best = best := by
  rw [innerTail166_16, scan, if_neg (by decide), innerCandidates166_16_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_17
    (if 3 ≤ innerCandidates166_16.length then search edge 3 [1009, 7027] innerCandidates166_16 8036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_17

def innerTail166_15 : List Nat := 6673 :: innerTail166_16
def innerCandidates166_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates166_15_checked : candidates edge 4 1009 6673 innerTail166_16 best.bound = innerCandidates166_15 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_15 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_15 best = best := by
  rw [innerTail166_15, scan, if_neg (by decide), innerCandidates166_15_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_16
    (if 3 ≤ innerCandidates166_15.length then search edge 3 [1009, 6673] innerCandidates166_15 7682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_16

def innerTail166_14 : List Nat := 6577 :: innerTail166_15
def innerCandidates166_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates166_14_checked : candidates edge 4 1009 6577 innerTail166_15 best.bound = innerCandidates166_14 := by
  unfold candidates
  have row : edge 6577 = SearchData.row6577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_14 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_14 best = best := by
  rw [innerTail166_14, scan, if_neg (by decide), innerCandidates166_14_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_15
    (if 3 ≤ innerCandidates166_14.length then search edge 3 [1009, 6577] innerCandidates166_14 7586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_15

def innerTail166_13 : List Nat := 6481 :: innerTail166_14
def innerCandidates166_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates166_13_checked : candidates edge 4 1009 6481 innerTail166_14 best.bound = innerCandidates166_13 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_13 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_13 best = best := by
  rw [innerTail166_13, scan, if_neg (by decide), innerCandidates166_13_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_14
    (if 3 ≤ innerCandidates166_13.length then search edge 3 [1009, 6481] innerCandidates166_13 7490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_14

def innerTail166_12 : List Nat := 6397 :: innerTail166_13
def innerCandidates166_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates166_12_checked : candidates edge 4 1009 6397 innerTail166_13 best.bound = innerCandidates166_12 := by
  unfold candidates
  have row : edge 6397 = SearchData.row6397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_12 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_12 best = best := by
  rw [innerTail166_12, scan, if_neg (by decide), innerCandidates166_12_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_13
    (if 3 ≤ innerCandidates166_12.length then search edge 3 [1009, 6397] innerCandidates166_12 7406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_13

def innerTail166_11 : List Nat := 6367 :: innerTail166_12
def innerCandidates166_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates166_11_checked : candidates edge 4 1009 6367 innerTail166_12 best.bound = innerCandidates166_11 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_11 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_11 best = best := by
  rw [innerTail166_11, scan, if_neg (by decide), innerCandidates166_11_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_12
    (if 3 ≤ innerCandidates166_11.length then search edge 3 [1009, 6367] innerCandidates166_11 7376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_12

def innerTail166_10 : List Nat := 6043 :: innerTail166_11
def innerCandidates166_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates166_10_checked : candidates edge 4 1009 6043 innerTail166_11 best.bound = innerCandidates166_10 := by
  unfold candidates
  have row : edge 6043 = SearchData.row6043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_10 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_10 best = best := by
  rw [innerTail166_10, scan, if_neg (by decide), innerCandidates166_10_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_11
    (if 3 ≤ innerCandidates166_10.length then search edge 3 [1009, 6043] innerCandidates166_10 7052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_11

def innerTail166_9 : List Nat := 5653 :: innerTail166_10
def innerCandidates166_9 : List Nat := [6367]
set_option maxRecDepth 4096 in
theorem innerCandidates166_9_checked : candidates edge 4 1009 5653 innerTail166_10 best.bound = innerCandidates166_9 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_9 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_9 best = best := by
  rw [innerTail166_9, scan, if_neg (by decide), innerCandidates166_9_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_10
    (if 3 ≤ innerCandidates166_9.length then search edge 3 [1009, 5653] innerCandidates166_9 6662 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_10

def innerTail166_8 : List Nat := 4657 :: innerTail166_9
def innerCandidates166_8 : List Nat := [7753]
set_option maxRecDepth 4096 in
theorem innerCandidates166_8_checked : candidates edge 4 1009 4657 innerTail166_9 best.bound = innerCandidates166_8 := by
  unfold candidates
  have row : edge 4657 = SearchData.row4657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_8 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_8 best = best := by
  rw [innerTail166_8, scan, if_neg (by decide), innerCandidates166_8_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_9
    (if 3 ≤ innerCandidates166_8.length then search edge 3 [1009, 4657] innerCandidates166_8 5666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_9

def innerTail166_7 : List Nat := 4099 :: innerTail166_8
def innerCandidates166_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates166_7_checked : candidates edge 4 1009 4099 innerTail166_8 best.bound = innerCandidates166_7 := by
  unfold candidates
  have row : edge 4099 = SearchData.row4099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_7 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_7 best = best := by
  rw [innerTail166_7, scan, if_neg (by decide), innerCandidates166_7_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_8
    (if 3 ≤ innerCandidates166_7.length then search edge 3 [1009, 4099] innerCandidates166_7 5108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_8

def innerTail166_6 : List Nat := 3823 :: innerTail166_7
def innerCandidates166_6 : List Nat := [7393]
set_option maxRecDepth 4096 in
theorem innerCandidates166_6_checked : candidates edge 4 1009 3823 innerTail166_7 best.bound = innerCandidates166_6 := by
  unfold candidates
  have row : edge 3823 = SearchData.row3823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_6 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_6 best = best := by
  rw [innerTail166_6, scan, if_neg (by decide), innerCandidates166_6_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_7
    (if 3 ≤ innerCandidates166_6.length then search edge 3 [1009, 3823] innerCandidates166_6 4832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_7

def innerTail166_5 : List Nat := 3613 :: innerTail166_6
def innerCandidates166_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates166_5_checked : candidates edge 4 1009 3613 innerTail166_6 best.bound = innerCandidates166_5 := by
  unfold candidates
  have row : edge 3613 = SearchData.row3613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_5 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_5 best = best := by
  rw [innerTail166_5, scan, if_neg (by decide), innerCandidates166_5_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_6
    (if 3 ≤ innerCandidates166_5.length then search edge 3 [1009, 3613] innerCandidates166_5 4622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_6

def innerTail166_4 : List Nat := 2851 :: innerTail166_5
def innerCandidates166_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates166_4_checked : candidates edge 4 1009 2851 innerTail166_5 best.bound = innerCandidates166_4 := by
  unfold candidates
  have row : edge 2851 = SearchData.row2851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_4 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_4 best = best := by
  rw [innerTail166_4, scan, if_neg (by decide), innerCandidates166_4_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_5
    (if 3 ≤ innerCandidates166_4.length then search edge 3 [1009, 2851] innerCandidates166_4 3860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_5

def innerTail166_3 : List Nat := 2473 :: innerTail166_4
def innerCandidates166_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates166_3_checked : candidates edge 4 1009 2473 innerTail166_4 best.bound = innerCandidates166_3 := by
  unfold candidates
  have row : edge 2473 = PrimePairCache.smallRow2473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_3 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_3 best = best := by
  rw [innerTail166_3, scan, if_neg (by decide), innerCandidates166_3_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_4
    (if 3 ≤ innerCandidates166_3.length then search edge 3 [1009, 2473] innerCandidates166_3 3482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_4

def innerTail166_2 : List Nat := 2221 :: innerTail166_3
def innerCandidates166_2 : List Nat := [6043, 6673]
set_option maxRecDepth 4096 in
theorem innerCandidates166_2_checked : candidates edge 4 1009 2221 innerTail166_3 best.bound = innerCandidates166_2 := by
  unfold candidates
  have row : edge 2221 = PrimePairCache.smallRow2221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_2 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_2 best = best := by
  rw [innerTail166_2, scan, if_neg (by decide), innerCandidates166_2_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_3
    (if 3 ≤ innerCandidates166_2.length then search edge 3 [1009, 2221] innerCandidates166_2 3230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_3

def innerTail166_1 : List Nat := 1933 :: innerTail166_2
def innerCandidates166_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates166_1_checked : candidates edge 4 1009 1933 innerTail166_2 best.bound = innerCandidates166_1 := by
  unfold candidates
  have row : edge 1933 = PrimePairCache.smallRow1933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_1 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_1 best = best := by
  rw [innerTail166_1, scan, if_neg (by decide), innerCandidates166_1_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_2
    (if 3 ≤ innerCandidates166_1.length then search edge 3 [1009, 1933] innerCandidates166_1 2942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_2

def innerTail166_0 : List Nat := 1789 :: innerTail166_1
def innerCandidates166_0 : List Nat := [7393]
set_option maxRecDepth 4096 in
theorem innerCandidates166_0_checked : candidates edge 4 1009 1789 innerTail166_1 best.bound = innerCandidates166_0 := by
  unfold candidates
  have row : edge 1789 = PrimePairCache.smallRow1789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan166_0 : scan edge (search edge 3) 4 [1009] 1009 innerTail166_0 best = best := by
  rw [innerTail166_0, scan, if_neg (by decide), innerCandidates166_0_checked]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_1
    (if 3 ≤ innerCandidates166_0.length then search edge 3 [1009, 1789] innerCandidates166_0 2798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan166_1

theorem rootBranch166 : search edge 4 [1009] rootCandidates166 1009 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1009] 1009 innerTail166_0 best = best
  exact innerScan166_0

def rootCandidates165 : List Nat := [1483, 1993, 2029, 2377, 2617, 3217, 3331, 4177, 4243, 4513, 4969, 5569, 6091, 6481, 6763, 7057, 7741]
set_option maxRecDepth 4096 in
theorem rootCandidates165_checked : candidates edge 5 0 997 tail166 best.bound = rootCandidates165 := by
  unfold candidates
  have row : edge 997 = PrimePairCache.smallRow997.contains := by funext q; rfl
  rw [row]
  decide

def innerTail165_17 : List Nat := []
theorem innerScan165_17 : scan edge (search edge 3) 4 [997] 997 innerTail165_17 best = best := rfl

def innerTail165_16 : List Nat := 7741 :: innerTail165_17
def innerCandidates165_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates165_16_checked : candidates edge 4 997 7741 innerTail165_17 best.bound = innerCandidates165_16 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan165_16 : scan edge (search edge 3) 4 [997] 997 innerTail165_16 best = best := by
  rw [innerTail165_16, scan, if_neg (by decide), innerCandidates165_16_checked]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_17
    (if 3 ≤ innerCandidates165_16.length then search edge 3 [997, 7741] innerCandidates165_16 8738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan165_17

def innerTail165_15 : List Nat := 7057 :: innerTail165_16
def innerCandidates165_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates165_15_checked : candidates edge 4 997 7057 innerTail165_16 best.bound = innerCandidates165_15 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan165_15 : scan edge (search edge 3) 4 [997] 997 innerTail165_15 best = best := by
  rw [innerTail165_15, scan, if_neg (by decide), innerCandidates165_15_checked]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_16
    (if 3 ≤ innerCandidates165_15.length then search edge 3 [997, 7057] innerCandidates165_15 8054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan165_16

def innerTail165_14 : List Nat := 6763 :: innerTail165_15
def innerCandidates165_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates165_14_checked : candidates edge 4 997 6763 innerTail165_15 best.bound = innerCandidates165_14 := by
  unfold candidates
  have row : edge 6763 = SearchData.row6763.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan165_14 : scan edge (search edge 3) 4 [997] 997 innerTail165_14 best = best := by
  rw [innerTail165_14, scan, if_neg (by decide), innerCandidates165_14_checked]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_15
    (if 3 ≤ innerCandidates165_14.length then search edge 3 [997, 6763] innerCandidates165_14 7760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan165_15

def innerTail165_13 : List Nat := 6481 :: innerTail165_14
def innerCandidates165_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates165_13_checked : candidates edge 4 997 6481 innerTail165_14 best.bound = innerCandidates165_13 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan165_13 : scan edge (search edge 3) 4 [997] 997 innerTail165_13 best = best := by
  rw [innerTail165_13, scan, if_neg (by decide), innerCandidates165_13_checked]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_14
    (if 3 ≤ innerCandidates165_13.length then search edge 3 [997, 6481] innerCandidates165_13 7478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan165_14

def innerTail165_12 : List Nat := 6091 :: innerTail165_13
def innerCandidates165_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates165_12_checked : candidates edge 4 997 6091 innerTail165_13 best.bound = innerCandidates165_12 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan165_12 : scan edge (search edge 3) 4 [997] 997 innerTail165_12 best = best := by
  rw [innerTail165_12, scan, if_neg (by decide), innerCandidates165_12_checked]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_13
    (if 3 ≤ innerCandidates165_12.length then search edge 3 [997, 6091] innerCandidates165_12 7088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan165_13

def innerTail165_11 : List Nat := 5569 :: innerTail165_12
def innerCandidates165_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates165_11_checked : candidates edge 4 997 5569 innerTail165_12 best.bound = innerCandidates165_11 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan165_11 : scan edge (search edge 3) 4 [997] 997 innerTail165_11 best = best := by
  rw [innerTail165_11, scan, if_neg (by decide), innerCandidates165_11_checked]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_12
    (if 3 ≤ innerCandidates165_11.length then search edge 3 [997, 5569] innerCandidates165_11 6566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan165_12

def innerTail165_10 : List Nat := 4969 :: innerTail165_11
def innerCandidates165_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates165_10_checked : candidates edge 4 997 4969 innerTail165_11 best.bound = innerCandidates165_10 := by
  unfold candidates
  have row : edge 4969 = SearchData.row4969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan165_10 : scan edge (search edge 3) 4 [997] 997 innerTail165_10 best = best := by
  rw [innerTail165_10, scan, if_neg (by decide), innerCandidates165_10_checked]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_11
    (if 3 ≤ innerCandidates165_10.length then search edge 3 [997, 4969] innerCandidates165_10 5966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan165_11

def innerTail165_9 : List Nat := 4513 :: innerTail165_10
def innerCandidates165_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates165_9_checked : candidates edge 4 997 4513 innerTail165_10 best.bound = innerCandidates165_9 := by
  unfold candidates
  have row : edge 4513 = SearchData.row4513.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan165_9 : scan edge (search edge 3) 4 [997] 997 innerTail165_9 best = best := by
  rw [innerTail165_9, scan, if_neg (by decide), innerCandidates165_9_checked]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_10
    (if 3 ≤ innerCandidates165_9.length then search edge 3 [997, 4513] innerCandidates165_9 5510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan165_10

def innerTail165_8 : List Nat := 4243 :: innerTail165_9
def innerCandidates165_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates165_8_checked : candidates edge 4 997 4243 innerTail165_9 best.bound = innerCandidates165_8 := by
  unfold candidates
  have row : edge 4243 = SearchData.row4243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan165_8 : scan edge (search edge 3) 4 [997] 997 innerTail165_8 best = best := by
  rw [innerTail165_8, scan, if_neg (by decide), innerCandidates165_8_checked]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_9
    (if 3 ≤ innerCandidates165_8.length then search edge 3 [997, 4243] innerCandidates165_8 5240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan165_9

def innerTail165_7 : List Nat := 4177 :: innerTail165_8
def innerCandidates165_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates165_7_checked : candidates edge 4 997 4177 innerTail165_8 best.bound = innerCandidates165_7 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan165_7 : scan edge (search edge 3) 4 [997] 997 innerTail165_7 best = best := by
  rw [innerTail165_7, scan, if_neg (by decide), innerCandidates165_7_checked]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_8
    (if 3 ≤ innerCandidates165_7.length then search edge 3 [997, 4177] innerCandidates165_7 5174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan165_8

def innerTail165_6 : List Nat := 3331 :: innerTail165_7
def innerCandidates165_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates165_6_checked : candidates edge 4 997 3331 innerTail165_7 best.bound = innerCandidates165_6 := by
  unfold candidates
  have row : edge 3331 = SearchData.row3331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan165_6 : scan edge (search edge 3) 4 [997] 997 innerTail165_6 best = best := by
  rw [innerTail165_6, scan, if_neg (by decide), innerCandidates165_6_checked]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_7
    (if 3 ≤ innerCandidates165_6.length then search edge 3 [997, 3331] innerCandidates165_6 4328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan165_7

def innerTail165_5 : List Nat := 3217 :: innerTail165_6
def innerCandidates165_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates165_5_checked : candidates edge 4 997 3217 innerTail165_6 best.bound = innerCandidates165_5 := by
  unfold candidates
  have row : edge 3217 = SearchData.row3217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan165_5 : scan edge (search edge 3) 4 [997] 997 innerTail165_5 best = best := by
  rw [innerTail165_5, scan, if_neg (by decide), innerCandidates165_5_checked]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_6
    (if 3 ≤ innerCandidates165_5.length then search edge 3 [997, 3217] innerCandidates165_5 4214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan165_6

def innerTail165_4 : List Nat := 2617 :: innerTail165_5
def innerCandidates165_4 : List Nat := [7741]
set_option maxRecDepth 4096 in
theorem innerCandidates165_4_checked : candidates edge 4 997 2617 innerTail165_5 best.bound = innerCandidates165_4 := by
  unfold candidates
  have row : edge 2617 = PrimePairCache.smallRow2617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan165_4 : scan edge (search edge 3) 4 [997] 997 innerTail165_4 best = best := by
  rw [innerTail165_4, scan, if_neg (by decide), innerCandidates165_4_checked]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_5
    (if 3 ≤ innerCandidates165_4.length then search edge 3 [997, 2617] innerCandidates165_4 3614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan165_5

def innerTail165_3 : List Nat := 2377 :: innerTail165_4
def innerCandidates165_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates165_3_checked : candidates edge 4 997 2377 innerTail165_4 best.bound = innerCandidates165_3 := by
  unfold candidates
  have row : edge 2377 = PrimePairCache.smallRow2377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan165_3 : scan edge (search edge 3) 4 [997] 997 innerTail165_3 best = best := by
  rw [innerTail165_3, scan, if_neg (by decide), innerCandidates165_3_checked]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_4
    (if 3 ≤ innerCandidates165_3.length then search edge 3 [997, 2377] innerCandidates165_3 3374 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan165_4

def innerTail165_2 : List Nat := 2029 :: innerTail165_3
def innerCandidates165_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates165_2_checked : candidates edge 4 997 2029 innerTail165_3 best.bound = innerCandidates165_2 := by
  unfold candidates
  have row : edge 2029 = PrimePairCache.smallRow2029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan165_2 : scan edge (search edge 3) 4 [997] 997 innerTail165_2 best = best := by
  rw [innerTail165_2, scan, if_neg (by decide), innerCandidates165_2_checked]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_3
    (if 3 ≤ innerCandidates165_2.length then search edge 3 [997, 2029] innerCandidates165_2 3026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan165_3

def innerTail165_1 : List Nat := 1993 :: innerTail165_2
def innerCandidates165_1 : List Nat := [6481]
set_option maxRecDepth 4096 in
theorem innerCandidates165_1_checked : candidates edge 4 997 1993 innerTail165_2 best.bound = innerCandidates165_1 := by
  unfold candidates
  have row : edge 1993 = PrimePairCache.smallRow1993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan165_1 : scan edge (search edge 3) 4 [997] 997 innerTail165_1 best = best := by
  rw [innerTail165_1, scan, if_neg (by decide), innerCandidates165_1_checked]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_2
    (if 3 ≤ innerCandidates165_1.length then search edge 3 [997, 1993] innerCandidates165_1 2990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan165_2

def innerTail165_0 : List Nat := 1483 :: innerTail165_1
def innerCandidates165_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates165_0_checked : candidates edge 4 997 1483 innerTail165_1 best.bound = innerCandidates165_0 := by
  unfold candidates
  have row : edge 1483 = PrimePairCache.smallRow1483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan165_0 : scan edge (search edge 3) 4 [997] 997 innerTail165_0 best = best := by
  rw [innerTail165_0, scan, if_neg (by decide), innerCandidates165_0_checked]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_1
    (if 3 ≤ innerCandidates165_0.length then search edge 3 [997, 1483] innerCandidates165_0 2480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan165_1

theorem rootBranch165 : search edge 4 [997] rootCandidates165 997 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [997] 997 innerTail165_0 best = best
  exact innerScan165_0

end ProjectEulerSolutions.PrimePairUpper.Search
