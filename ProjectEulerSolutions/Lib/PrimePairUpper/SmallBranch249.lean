import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates18 : List Nat := [277, 547, 571, 607, 643, 757, 823, 1117, 1447, 1597, 1933, 2029, 2287, 2293, 2467, 2521, 2617, 2689, 2749, 3391, 3433, 3517, 3823, 3847, 4177, 4549, 4663, 4759, 4813, 4957, 5107, 5443, 5659, 5821, 5953, 6091, 6121, 6277, 6367, 6691, 6793, 6823, 7129, 7159, 7207, 7309, 7351, 7537, 7591, 7687]
set_option maxRecDepth 4096 in
theorem rootCandidates18_checked : candidates edge 5 0 73 tail19 best.bound = rootCandidates18 := by
  unfold candidates
  have row : edge 73 = PrimePairCache.smallRow73.contains := by funext q; rfl
  rw [row]
  decide

def innerTail18_50 : List Nat := []
theorem innerScan18_50 : scan edge (search edge 3) 4 [73] 73 innerTail18_50 best = best := rfl

def innerTail18_49 : List Nat := 7687 :: innerTail18_50
def innerCandidates18_49 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_49_checked : candidates edge 4 73 7687 innerTail18_50 best.bound = innerCandidates18_49 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_49 : scan edge (search edge 3) 4 [73] 73 innerTail18_49 best = best := by
  rw [innerTail18_49, scan, if_neg (by decide), innerCandidates18_49_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_50
    (if 3 ≤ innerCandidates18_49.length then search edge 3 [73, 7687] innerCandidates18_49 7760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_50

def innerTail18_48 : List Nat := 7591 :: innerTail18_49
def innerCandidates18_48 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_48_checked : candidates edge 4 73 7591 innerTail18_49 best.bound = innerCandidates18_48 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_48 : scan edge (search edge 3) 4 [73] 73 innerTail18_48 best = best := by
  rw [innerTail18_48, scan, if_neg (by decide), innerCandidates18_48_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_49
    (if 3 ≤ innerCandidates18_48.length then search edge 3 [73, 7591] innerCandidates18_48 7664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_49

def innerTail18_47 : List Nat := 7537 :: innerTail18_48
def innerCandidates18_47 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_47_checked : candidates edge 4 73 7537 innerTail18_48 best.bound = innerCandidates18_47 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_47 : scan edge (search edge 3) 4 [73] 73 innerTail18_47 best = best := by
  rw [innerTail18_47, scan, if_neg (by decide), innerCandidates18_47_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_48
    (if 3 ≤ innerCandidates18_47.length then search edge 3 [73, 7537] innerCandidates18_47 7610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_48

def innerTail18_46 : List Nat := 7351 :: innerTail18_47
def innerCandidates18_46 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_46_checked : candidates edge 4 73 7351 innerTail18_47 best.bound = innerCandidates18_46 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_46 : scan edge (search edge 3) 4 [73] 73 innerTail18_46 best = best := by
  rw [innerTail18_46, scan, if_neg (by decide), innerCandidates18_46_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_47
    (if 3 ≤ innerCandidates18_46.length then search edge 3 [73, 7351] innerCandidates18_46 7424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_47

def innerTail18_45 : List Nat := 7309 :: innerTail18_46
def innerCandidates18_45 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_45_checked : candidates edge 4 73 7309 innerTail18_46 best.bound = innerCandidates18_45 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_45 : scan edge (search edge 3) 4 [73] 73 innerTail18_45 best = best := by
  rw [innerTail18_45, scan, if_neg (by decide), innerCandidates18_45_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_46
    (if 3 ≤ innerCandidates18_45.length then search edge 3 [73, 7309] innerCandidates18_45 7382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_46

def innerTail18_44 : List Nat := 7207 :: innerTail18_45
def innerCandidates18_44 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_44_checked : candidates edge 4 73 7207 innerTail18_45 best.bound = innerCandidates18_44 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_44 : scan edge (search edge 3) 4 [73] 73 innerTail18_44 best = best := by
  rw [innerTail18_44, scan, if_neg (by decide), innerCandidates18_44_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_45
    (if 3 ≤ innerCandidates18_44.length then search edge 3 [73, 7207] innerCandidates18_44 7280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_45

def innerTail18_43 : List Nat := 7159 :: innerTail18_44
def innerCandidates18_43 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_43_checked : candidates edge 4 73 7159 innerTail18_44 best.bound = innerCandidates18_43 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_43 : scan edge (search edge 3) 4 [73] 73 innerTail18_43 best = best := by
  rw [innerTail18_43, scan, if_neg (by decide), innerCandidates18_43_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_44
    (if 3 ≤ innerCandidates18_43.length then search edge 3 [73, 7159] innerCandidates18_43 7232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_44

def innerTail18_42 : List Nat := 7129 :: innerTail18_43
def innerCandidates18_42 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_42_checked : candidates edge 4 73 7129 innerTail18_43 best.bound = innerCandidates18_42 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_42 : scan edge (search edge 3) 4 [73] 73 innerTail18_42 best = best := by
  rw [innerTail18_42, scan, if_neg (by decide), innerCandidates18_42_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_43
    (if 3 ≤ innerCandidates18_42.length then search edge 3 [73, 7129] innerCandidates18_42 7202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_43

def innerTail18_41 : List Nat := 6823 :: innerTail18_42
def innerCandidates18_41 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_41_checked : candidates edge 4 73 6823 innerTail18_42 best.bound = innerCandidates18_41 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_41 : scan edge (search edge 3) 4 [73] 73 innerTail18_41 best = best := by
  rw [innerTail18_41, scan, if_neg (by decide), innerCandidates18_41_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_42
    (if 3 ≤ innerCandidates18_41.length then search edge 3 [73, 6823] innerCandidates18_41 6896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_42

def innerTail18_40 : List Nat := 6793 :: innerTail18_41
def innerCandidates18_40 : List Nat := [7159]
set_option maxRecDepth 4096 in
theorem innerCandidates18_40_checked : candidates edge 4 73 6793 innerTail18_41 best.bound = innerCandidates18_40 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_40 : scan edge (search edge 3) 4 [73] 73 innerTail18_40 best = best := by
  rw [innerTail18_40, scan, if_neg (by decide), innerCandidates18_40_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_41
    (if 3 ≤ innerCandidates18_40.length then search edge 3 [73, 6793] innerCandidates18_40 6866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_41

def innerTail18_39 : List Nat := 6691 :: innerTail18_40
def innerCandidates18_39 : List Nat := [7687]
set_option maxRecDepth 4096 in
theorem innerCandidates18_39_checked : candidates edge 4 73 6691 innerTail18_40 best.bound = innerCandidates18_39 := by
  unfold candidates
  have row : edge 6691 = SearchData.row6691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_39 : scan edge (search edge 3) 4 [73] 73 innerTail18_39 best = best := by
  rw [innerTail18_39, scan, if_neg (by decide), innerCandidates18_39_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_40
    (if 3 ≤ innerCandidates18_39.length then search edge 3 [73, 6691] innerCandidates18_39 6764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_40

def innerTail18_38 : List Nat := 6367 :: innerTail18_39
def innerCandidates18_38 : List Nat := [6793]
set_option maxRecDepth 4096 in
theorem innerCandidates18_38_checked : candidates edge 4 73 6367 innerTail18_39 best.bound = innerCandidates18_38 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_38 : scan edge (search edge 3) 4 [73] 73 innerTail18_38 best = best := by
  rw [innerTail18_38, scan, if_neg (by decide), innerCandidates18_38_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_39
    (if 3 ≤ innerCandidates18_38.length then search edge 3 [73, 6367] innerCandidates18_38 6440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_39

def innerTail18_37 : List Nat := 6277 :: innerTail18_38
def innerCandidates18_37 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_37_checked : candidates edge 4 73 6277 innerTail18_38 best.bound = innerCandidates18_37 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_37 : scan edge (search edge 3) 4 [73] 73 innerTail18_37 best = best := by
  rw [innerTail18_37, scan, if_neg (by decide), innerCandidates18_37_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_38
    (if 3 ≤ innerCandidates18_37.length then search edge 3 [73, 6277] innerCandidates18_37 6350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_38

def innerTail18_36 : List Nat := 6121 :: innerTail18_37
def innerCandidates18_36 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_36_checked : candidates edge 4 73 6121 innerTail18_37 best.bound = innerCandidates18_36 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_36 : scan edge (search edge 3) 4 [73] 73 innerTail18_36 best = best := by
  rw [innerTail18_36, scan, if_neg (by decide), innerCandidates18_36_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_37
    (if 3 ≤ innerCandidates18_36.length then search edge 3 [73, 6121] innerCandidates18_36 6194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_37

def innerTail18_35 : List Nat := 6091 :: innerTail18_36
def innerCandidates18_35 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_35_checked : candidates edge 4 73 6091 innerTail18_36 best.bound = innerCandidates18_35 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_35 : scan edge (search edge 3) 4 [73] 73 innerTail18_35 best = best := by
  rw [innerTail18_35, scan, if_neg (by decide), innerCandidates18_35_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_36
    (if 3 ≤ innerCandidates18_35.length then search edge 3 [73, 6091] innerCandidates18_35 6164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_36

def innerTail18_34 : List Nat := 5953 :: innerTail18_35
def innerCandidates18_34 : List Nat := [6277]
set_option maxRecDepth 4096 in
theorem innerCandidates18_34_checked : candidates edge 4 73 5953 innerTail18_35 best.bound = innerCandidates18_34 := by
  unfold candidates
  have row : edge 5953 = SearchData.row5953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_34 : scan edge (search edge 3) 4 [73] 73 innerTail18_34 best = best := by
  rw [innerTail18_34, scan, if_neg (by decide), innerCandidates18_34_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_35
    (if 3 ≤ innerCandidates18_34.length then search edge 3 [73, 5953] innerCandidates18_34 6026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_35

def innerTail18_33 : List Nat := 5821 :: innerTail18_34
def innerCandidates18_33 : List Nat := [6121]
set_option maxRecDepth 4096 in
theorem innerCandidates18_33_checked : candidates edge 4 73 5821 innerTail18_34 best.bound = innerCandidates18_33 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_33 : scan edge (search edge 3) 4 [73] 73 innerTail18_33 best = best := by
  rw [innerTail18_33, scan, if_neg (by decide), innerCandidates18_33_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_34
    (if 3 ≤ innerCandidates18_33.length then search edge 3 [73, 5821] innerCandidates18_33 5894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_34

def innerTail18_32 : List Nat := 5659 :: innerTail18_33
def innerCandidates18_32 : List Nat := [5953, 6121]
set_option maxRecDepth 4096 in
theorem innerCandidates18_32_checked : candidates edge 4 73 5659 innerTail18_33 best.bound = innerCandidates18_32 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_32 : scan edge (search edge 3) 4 [73] 73 innerTail18_32 best = best := by
  rw [innerTail18_32, scan, if_neg (by decide), innerCandidates18_32_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_33
    (if 3 ≤ innerCandidates18_32.length then search edge 3 [73, 5659] innerCandidates18_32 5732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_33

def innerTail18_31 : List Nat := 5443 :: innerTail18_32
def innerCandidates18_31 : List Nat := [7351]
set_option maxRecDepth 4096 in
theorem innerCandidates18_31_checked : candidates edge 4 73 5443 innerTail18_32 best.bound = innerCandidates18_31 := by
  unfold candidates
  have row : edge 5443 = SearchData.row5443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_31 : scan edge (search edge 3) 4 [73] 73 innerTail18_31 best = best := by
  rw [innerTail18_31, scan, if_neg (by decide), innerCandidates18_31_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_32
    (if 3 ≤ innerCandidates18_31.length then search edge 3 [73, 5443] innerCandidates18_31 5516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_32

def innerTail18_30 : List Nat := 5107 :: innerTail18_31
def innerCandidates18_30 : List Nat := [6793]
set_option maxRecDepth 4096 in
theorem innerCandidates18_30_checked : candidates edge 4 73 5107 innerTail18_31 best.bound = innerCandidates18_30 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_30 : scan edge (search edge 3) 4 [73] 73 innerTail18_30 best = best := by
  rw [innerTail18_30, scan, if_neg (by decide), innerCandidates18_30_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_31
    (if 3 ≤ innerCandidates18_30.length then search edge 3 [73, 5107] innerCandidates18_30 5180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_31

def innerTail18_29 : List Nat := 4957 :: innerTail18_30
def innerCandidates18_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_29_checked : candidates edge 4 73 4957 innerTail18_30 best.bound = innerCandidates18_29 := by
  unfold candidates
  have row : edge 4957 = SearchData.row4957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_29 : scan edge (search edge 3) 4 [73] 73 innerTail18_29 best = best := by
  rw [innerTail18_29, scan, if_neg (by decide), innerCandidates18_29_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_30
    (if 3 ≤ innerCandidates18_29.length then search edge 3 [73, 4957] innerCandidates18_29 5030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_30

def innerTail18_28 : List Nat := 4813 :: innerTail18_29
def innerCandidates18_28 : List Nat := [7351]
set_option maxRecDepth 4096 in
theorem innerCandidates18_28_checked : candidates edge 4 73 4813 innerTail18_29 best.bound = innerCandidates18_28 := by
  unfold candidates
  have row : edge 4813 = SearchData.row4813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_28 : scan edge (search edge 3) 4 [73] 73 innerTail18_28 best = best := by
  rw [innerTail18_28, scan, if_neg (by decide), innerCandidates18_28_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_29
    (if 3 ≤ innerCandidates18_28.length then search edge 3 [73, 4813] innerCandidates18_28 4886 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_29

def innerTail18_27 : List Nat := 4759 :: innerTail18_28
def innerCandidates18_27 : List Nat := [6823]
set_option maxRecDepth 4096 in
theorem innerCandidates18_27_checked : candidates edge 4 73 4759 innerTail18_28 best.bound = innerCandidates18_27 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_27 : scan edge (search edge 3) 4 [73] 73 innerTail18_27 best = best := by
  rw [innerTail18_27, scan, if_neg (by decide), innerCandidates18_27_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_28
    (if 3 ≤ innerCandidates18_27.length then search edge 3 [73, 4759] innerCandidates18_27 4832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_28

def innerTail18_26 : List Nat := 4663 :: innerTail18_27
def innerCandidates18_26 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_26_checked : candidates edge 4 73 4663 innerTail18_27 best.bound = innerCandidates18_26 := by
  unfold candidates
  have row : edge 4663 = SearchData.row4663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_26 : scan edge (search edge 3) 4 [73] 73 innerTail18_26 best = best := by
  rw [innerTail18_26, scan, if_neg (by decide), innerCandidates18_26_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_27
    (if 3 ≤ innerCandidates18_26.length then search edge 3 [73, 4663] innerCandidates18_26 4736 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_27

def innerTail18_25 : List Nat := 4549 :: innerTail18_26
def innerCandidates18_25 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_25_checked : candidates edge 4 73 4549 innerTail18_26 best.bound = innerCandidates18_25 := by
  unfold candidates
  have row : edge 4549 = SearchData.row4549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_25 : scan edge (search edge 3) 4 [73] 73 innerTail18_25 best = best := by
  rw [innerTail18_25, scan, if_neg (by decide), innerCandidates18_25_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_26
    (if 3 ≤ innerCandidates18_25.length then search edge 3 [73, 4549] innerCandidates18_25 4622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_26

def innerTail18_24 : List Nat := 4177 :: innerTail18_25
def innerCandidates18_24 : List Nat := [7537]
set_option maxRecDepth 4096 in
theorem innerCandidates18_24_checked : candidates edge 4 73 4177 innerTail18_25 best.bound = innerCandidates18_24 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_24 : scan edge (search edge 3) 4 [73] 73 innerTail18_24 best = best := by
  rw [innerTail18_24, scan, if_neg (by decide), innerCandidates18_24_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_25
    (if 3 ≤ innerCandidates18_24.length then search edge 3 [73, 4177] innerCandidates18_24 4250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_25

def innerTail18_23 : List Nat := 3847 :: innerTail18_24
def innerCandidates18_23 : List Nat := [5659, 6367]
set_option maxRecDepth 4096 in
theorem innerCandidates18_23_checked : candidates edge 4 73 3847 innerTail18_24 best.bound = innerCandidates18_23 := by
  unfold candidates
  have row : edge 3847 = SearchData.row3847.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_23 : scan edge (search edge 3) 4 [73] 73 innerTail18_23 best = best := by
  rw [innerTail18_23, scan, if_neg (by decide), innerCandidates18_23_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_24
    (if 3 ≤ innerCandidates18_23.length then search edge 3 [73, 3847] innerCandidates18_23 3920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_24

def innerTail18_22 : List Nat := 3823 :: innerTail18_23
def innerCandidates18_22 : List Nat := [4549, 4759]
set_option maxRecDepth 4096 in
theorem innerCandidates18_22_checked : candidates edge 4 73 3823 innerTail18_23 best.bound = innerCandidates18_22 := by
  unfold candidates
  have row : edge 3823 = SearchData.row3823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_22 : scan edge (search edge 3) 4 [73] 73 innerTail18_22 best = best := by
  rw [innerTail18_22, scan, if_neg (by decide), innerCandidates18_22_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_23
    (if 3 ≤ innerCandidates18_22.length then search edge 3 [73, 3823] innerCandidates18_22 3896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_23

def innerTail18_21 : List Nat := 3517 :: innerTail18_22
def innerCandidates18_21 : List Nat := [4177, 5659, 7159]
set_option maxRecDepth 4096 in
theorem innerCandidates18_21_checked : candidates edge 4 73 3517 innerTail18_22 best.bound = innerCandidates18_21 := by
  unfold candidates
  have row : edge 3517 = SearchData.row3517.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild18_21 : search edge 3 [73, 3517] innerCandidates18_21 3590 best = best := by decide

theorem innerScan18_21 : scan edge (search edge 3) 4 [73] 73 innerTail18_21 best = best := by
  rw [innerTail18_21, scan, if_neg (by decide), innerCandidates18_21_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_22
    (if 3 ≤ innerCandidates18_21.length then search edge 3 [73, 3517] innerCandidates18_21 3590 best else best) = best
  rw [if_pos (by decide), innerChild18_21]
  exact innerScan18_22

def innerTail18_20 : List Nat := 3433 :: innerTail18_21
def innerCandidates18_20 : List Nat := [7129, 7207]
set_option maxRecDepth 4096 in
theorem innerCandidates18_20_checked : candidates edge 4 73 3433 innerTail18_21 best.bound = innerCandidates18_20 := by
  unfold candidates
  have row : edge 3433 = SearchData.row3433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_20 : scan edge (search edge 3) 4 [73] 73 innerTail18_20 best = best := by
  rw [innerTail18_20, scan, if_neg (by decide), innerCandidates18_20_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_21
    (if 3 ≤ innerCandidates18_20.length then search edge 3 [73, 3433] innerCandidates18_20 3506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_21

def innerTail18_19 : List Nat := 3391 :: innerTail18_20
def innerCandidates18_19 : List Nat := [3433, 5107, 5443]
set_option maxRecDepth 4096 in
theorem innerCandidates18_19_checked : candidates edge 4 73 3391 innerTail18_20 best.bound = innerCandidates18_19 := by
  unfold candidates
  have row : edge 3391 = SearchData.row3391.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild18_19 : search edge 3 [73, 3391] innerCandidates18_19 3464 best = best := by decide

theorem innerScan18_19 : scan edge (search edge 3) 4 [73] 73 innerTail18_19 best = best := by
  rw [innerTail18_19, scan, if_neg (by decide), innerCandidates18_19_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_20
    (if 3 ≤ innerCandidates18_19.length then search edge 3 [73, 3391] innerCandidates18_19 3464 best else best) = best
  rw [if_pos (by decide), innerChild18_19]
  exact innerScan18_20

def innerTail18_18 : List Nat := 2749 :: innerTail18_19
def innerCandidates18_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_18_checked : candidates edge 4 73 2749 innerTail18_19 best.bound = innerCandidates18_18 := by
  unfold candidates
  have row : edge 2749 = SearchData.row2749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_18 : scan edge (search edge 3) 4 [73] 73 innerTail18_18 best = best := by
  rw [innerTail18_18, scan, if_neg (by decide), innerCandidates18_18_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_19
    (if 3 ≤ innerCandidates18_18.length then search edge 3 [73, 2749] innerCandidates18_18 2822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_19

def innerTail18_17 : List Nat := 2689 :: innerTail18_18
def innerCandidates18_17 : List Nat := [6367]
set_option maxRecDepth 4096 in
theorem innerCandidates18_17_checked : candidates edge 4 73 2689 innerTail18_18 best.bound = innerCandidates18_17 := by
  unfold candidates
  have row : edge 2689 = SearchData.row2689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_17 : scan edge (search edge 3) 4 [73] 73 innerTail18_17 best = best := by
  rw [innerTail18_17, scan, if_neg (by decide), innerCandidates18_17_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_18
    (if 3 ≤ innerCandidates18_17.length then search edge 3 [73, 2689] innerCandidates18_17 2762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_18

def innerTail18_16 : List Nat := 2617 :: innerTail18_17
def innerCandidates18_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_16_checked : candidates edge 4 73 2617 innerTail18_17 best.bound = innerCandidates18_16 := by
  unfold candidates
  have row : edge 2617 = PrimePairCache.smallRow2617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_16 : scan edge (search edge 3) 4 [73] 73 innerTail18_16 best = best := by
  rw [innerTail18_16, scan, if_neg (by decide), innerCandidates18_16_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_17
    (if 3 ≤ innerCandidates18_16.length then search edge 3 [73, 2617] innerCandidates18_16 2690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_17

def innerTail18_15 : List Nat := 2521 :: innerTail18_16
def innerCandidates18_15 : List Nat := [4759]
set_option maxRecDepth 4096 in
theorem innerCandidates18_15_checked : candidates edge 4 73 2521 innerTail18_16 best.bound = innerCandidates18_15 := by
  unfold candidates
  have row : edge 2521 = PrimePairCache.smallRow2521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_15 : scan edge (search edge 3) 4 [73] 73 innerTail18_15 best = best := by
  rw [innerTail18_15, scan, if_neg (by decide), innerCandidates18_15_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_16
    (if 3 ≤ innerCandidates18_15.length then search edge 3 [73, 2521] innerCandidates18_15 2594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_16

def innerTail18_14 : List Nat := 2467 :: innerTail18_15
def innerCandidates18_14 : List Nat := [6277]
set_option maxRecDepth 4096 in
theorem innerCandidates18_14_checked : candidates edge 4 73 2467 innerTail18_15 best.bound = innerCandidates18_14 := by
  unfold candidates
  have row : edge 2467 = PrimePairCache.smallRow2467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_14 : scan edge (search edge 3) 4 [73] 73 innerTail18_14 best = best := by
  rw [innerTail18_14, scan, if_neg (by decide), innerCandidates18_14_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_15
    (if 3 ≤ innerCandidates18_14.length then search edge 3 [73, 2467] innerCandidates18_14 2540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_15

def innerTail18_13 : List Nat := 2293 :: innerTail18_14
def innerCandidates18_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_13_checked : candidates edge 4 73 2293 innerTail18_14 best.bound = innerCandidates18_13 := by
  unfold candidates
  have row : edge 2293 = PrimePairCache.smallRow2293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_13 : scan edge (search edge 3) 4 [73] 73 innerTail18_13 best = best := by
  rw [innerTail18_13, scan, if_neg (by decide), innerCandidates18_13_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_14
    (if 3 ≤ innerCandidates18_13.length then search edge 3 [73, 2293] innerCandidates18_13 2366 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_14

def innerTail18_12 : List Nat := 2287 :: innerTail18_13
def innerCandidates18_12 : List Nat := [6277, 6793, 7537]
set_option maxRecDepth 4096 in
theorem innerCandidates18_12_checked : candidates edge 4 73 2287 innerTail18_13 best.bound = innerCandidates18_12 := by
  unfold candidates
  have row : edge 2287 = PrimePairCache.smallRow2287.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild18_12 : search edge 3 [73, 2287] innerCandidates18_12 2360 best = best := by decide

theorem innerScan18_12 : scan edge (search edge 3) 4 [73] 73 innerTail18_12 best = best := by
  rw [innerTail18_12, scan, if_neg (by decide), innerCandidates18_12_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_13
    (if 3 ≤ innerCandidates18_12.length then search edge 3 [73, 2287] innerCandidates18_12 2360 best else best) = best
  rw [if_pos (by decide), innerChild18_12]
  exact innerScan18_13

def innerTail18_11 : List Nat := 2029 :: innerTail18_12
def innerCandidates18_11 : List Nat := [6121, 7309]
set_option maxRecDepth 4096 in
theorem innerCandidates18_11_checked : candidates edge 4 73 2029 innerTail18_12 best.bound = innerCandidates18_11 := by
  unfold candidates
  have row : edge 2029 = PrimePairCache.smallRow2029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_11 : scan edge (search edge 3) 4 [73] 73 innerTail18_11 best = best := by
  rw [innerTail18_11, scan, if_neg (by decide), innerCandidates18_11_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_12
    (if 3 ≤ innerCandidates18_11.length then search edge 3 [73, 2029] innerCandidates18_11 2102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_12

def innerTail18_10 : List Nat := 1933 :: innerTail18_11
def innerCandidates18_10 : List Nat := [2293]
set_option maxRecDepth 4096 in
theorem innerCandidates18_10_checked : candidates edge 4 73 1933 innerTail18_11 best.bound = innerCandidates18_10 := by
  unfold candidates
  have row : edge 1933 = PrimePairCache.smallRow1933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_10 : scan edge (search edge 3) 4 [73] 73 innerTail18_10 best = best := by
  rw [innerTail18_10, scan, if_neg (by decide), innerCandidates18_10_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_11
    (if 3 ≤ innerCandidates18_10.length then search edge 3 [73, 1933] innerCandidates18_10 2006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_11

def innerTail18_9 : List Nat := 1597 :: innerTail18_10
def innerCandidates18_9 : List Nat := [2617, 3823]
set_option maxRecDepth 4096 in
theorem innerCandidates18_9_checked : candidates edge 4 73 1597 innerTail18_10 best.bound = innerCandidates18_9 := by
  unfold candidates
  have row : edge 1597 = PrimePairCache.smallRow1597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_9 : scan edge (search edge 3) 4 [73] 73 innerTail18_9 best = best := by
  rw [innerTail18_9, scan, if_neg (by decide), innerCandidates18_9_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_10
    (if 3 ≤ innerCandidates18_9.length then search edge 3 [73, 1597] innerCandidates18_9 1670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_10

def innerTail18_8 : List Nat := 1447 :: innerTail18_9
def innerCandidates18_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_8_checked : candidates edge 4 73 1447 innerTail18_9 best.bound = innerCandidates18_8 := by
  unfold candidates
  have row : edge 1447 = PrimePairCache.smallRow1447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_8 : scan edge (search edge 3) 4 [73] 73 innerTail18_8 best = best := by
  rw [innerTail18_8, scan, if_neg (by decide), innerCandidates18_8_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_9
    (if 3 ≤ innerCandidates18_8.length then search edge 3 [73, 1447] innerCandidates18_8 1520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_9

def innerTail18_7 : List Nat := 1117 :: innerTail18_8
def innerCandidates18_7 : List Nat := [3823, 7591]
set_option maxRecDepth 4096 in
theorem innerCandidates18_7_checked : candidates edge 4 73 1117 innerTail18_8 best.bound = innerCandidates18_7 := by
  unfold candidates
  have row : edge 1117 = PrimePairCache.smallRow1117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_7 : scan edge (search edge 3) 4 [73] 73 innerTail18_7 best = best := by
  rw [innerTail18_7, scan, if_neg (by decide), innerCandidates18_7_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_8
    (if 3 ≤ innerCandidates18_7.length then search edge 3 [73, 1117] innerCandidates18_7 1190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_8

def innerTail18_6 : List Nat := 823 :: innerTail18_7
def innerCandidates18_6 : List Nat := [1933]
set_option maxRecDepth 4096 in
theorem innerCandidates18_6_checked : candidates edge 4 73 823 innerTail18_7 best.bound = innerCandidates18_6 := by
  unfold candidates
  have row : edge 823 = PrimePairCache.smallRow823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_6 : scan edge (search edge 3) 4 [73] 73 innerTail18_6 best = best := by
  rw [innerTail18_6, scan, if_neg (by decide), innerCandidates18_6_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_7
    (if 3 ≤ innerCandidates18_6.length then search edge 3 [73, 823] innerCandidates18_6 896 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_7

def innerTail18_5 : List Nat := 757 :: innerTail18_6
def innerCandidates18_5 : List Nat := [4549]
set_option maxRecDepth 4096 in
theorem innerCandidates18_5_checked : candidates edge 4 73 757 innerTail18_6 best.bound = innerCandidates18_5 := by
  unfold candidates
  have row : edge 757 = PrimePairCache.smallRow757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_5 : scan edge (search edge 3) 4 [73] 73 innerTail18_5 best = best := by
  rw [innerTail18_5, scan, if_neg (by decide), innerCandidates18_5_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_6
    (if 3 ≤ innerCandidates18_5.length then search edge 3 [73, 757] innerCandidates18_5 830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_6

def innerTail18_4 : List Nat := 643 :: innerTail18_5
def innerCandidates18_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates18_4_checked : candidates edge 4 73 643 innerTail18_5 best.bound = innerCandidates18_4 := by
  unfold candidates
  have row : edge 643 = PrimePairCache.smallRow643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_4 : scan edge (search edge 3) 4 [73] 73 innerTail18_4 best = best := by
  rw [innerTail18_4, scan, if_neg (by decide), innerCandidates18_4_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_5
    (if 3 ≤ innerCandidates18_4.length then search edge 3 [73, 643] innerCandidates18_4 716 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_5

def innerTail18_3 : List Nat := 607 :: innerTail18_4
def innerCandidates18_3 : List Nat := [2467, 5107, 6121]
set_option maxRecDepth 4096 in
theorem innerCandidates18_3_checked : candidates edge 4 73 607 innerTail18_4 best.bound = innerCandidates18_3 := by
  unfold candidates
  have row : edge 607 = PrimePairCache.smallRow607.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild18_3 : search edge 3 [73, 607] innerCandidates18_3 680 best = best := by decide

theorem innerScan18_3 : scan edge (search edge 3) 4 [73] 73 innerTail18_3 best = best := by
  rw [innerTail18_3, scan, if_neg (by decide), innerCandidates18_3_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_4
    (if 3 ≤ innerCandidates18_3.length then search edge 3 [73, 607] innerCandidates18_3 680 best else best) = best
  rw [if_pos (by decide), innerChild18_3]
  exact innerScan18_4

def innerTail18_2 : List Nat := 571 :: innerTail18_3
def innerCandidates18_2 : List Nat := [1933, 4759]
set_option maxRecDepth 4096 in
theorem innerCandidates18_2_checked : candidates edge 4 73 571 innerTail18_3 best.bound = innerCandidates18_2 := by
  unfold candidates
  have row : edge 571 = PrimePairCache.smallRow571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_2 : scan edge (search edge 3) 4 [73] 73 innerTail18_2 best = best := by
  rw [innerTail18_2, scan, if_neg (by decide), innerCandidates18_2_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_3
    (if 3 ≤ innerCandidates18_2.length then search edge 3 [73, 571] innerCandidates18_2 644 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_3

def innerTail18_1 : List Nat := 547 :: innerTail18_2
def innerCandidates18_1 : List Nat := [643, 823, 4549, 5443]
set_option maxRecDepth 4096 in
theorem innerCandidates18_1_checked : candidates edge 4 73 547 innerTail18_2 best.bound = innerCandidates18_1 := by
  unfold candidates
  have row : edge 547 = PrimePairCache.smallRow547.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild18_1 : search edge 3 [73, 547] innerCandidates18_1 620 best = best := by decide

theorem innerScan18_1 : scan edge (search edge 3) 4 [73] 73 innerTail18_1 best = best := by
  rw [innerTail18_1, scan, if_neg (by decide), innerCandidates18_1_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_2
    (if 3 ≤ innerCandidates18_1.length then search edge 3 [73, 547] innerCandidates18_1 620 best else best) = best
  rw [if_pos (by decide), innerChild18_1]
  exact innerScan18_2

def innerTail18_0 : List Nat := 277 :: innerTail18_1
def innerCandidates18_0 : List Nat := [1933]
set_option maxRecDepth 4096 in
theorem innerCandidates18_0_checked : candidates edge 4 73 277 innerTail18_1 best.bound = innerCandidates18_0 := by
  unfold candidates
  have row : edge 277 = PrimePairCache.smallRow277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan18_0 : scan edge (search edge 3) 4 [73] 73 innerTail18_0 best = best := by
  rw [innerTail18_0, scan, if_neg (by decide), innerCandidates18_0_checked]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_1
    (if 3 ≤ innerCandidates18_0.length then search edge 3 [73, 277] innerCandidates18_0 350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan18_1

theorem rootBranch18 : search edge 4 [73] rootCandidates18 73 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [73] 73 innerTail18_0 best = best
  exact innerScan18_0

end ProjectEulerSolutions.PrimePairUpper.Search
