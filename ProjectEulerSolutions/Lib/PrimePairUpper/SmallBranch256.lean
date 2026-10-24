import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates11 : List Nat := [97, 103, 223, 271, 499, 613, 691, 1051, 1153, 1213, 1381, 1399, 1621, 1693, 1777, 1867, 1993, 2251, 2389, 2503, 2557, 2749, 3049, 3187, 3253, 3259, 3319, 3361, 3373, 3469, 3877, 3931, 4111, 4243, 4363, 4783, 4813, 5563, 5569, 5653, 5857, 5923, 6091, 6343, 6529, 6547, 6967, 7159, 7351, 7753, 8287, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates11_checked : candidates edge 5 0 43 tail12 best.bound = rootCandidates11 := by
  unfold candidates
  have row : edge 43 = PrimePairCache.smallRow43.contains := by funext q; rfl
  rw [row]
  decide

def innerTail11_52 : List Nat := []
theorem innerScan11_52 : scan edge (search edge 3) 4 [43] 43 innerTail11_52 best = best := rfl

def innerTail11_51 : List Nat := 8329 :: innerTail11_52
def innerCandidates11_51 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_51_checked : candidates edge 4 43 8329 innerTail11_52 best.bound = innerCandidates11_51 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_51 : scan edge (search edge 3) 4 [43] 43 innerTail11_51 best = best := by
  rw [innerTail11_51, scan, if_neg (by decide), innerCandidates11_51_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_52
    (if 3 ≤ innerCandidates11_51.length then search edge 3 [43, 8329] innerCandidates11_51 8372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_52

def innerTail11_50 : List Nat := 8287 :: innerTail11_51
def innerCandidates11_50 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_50_checked : candidates edge 4 43 8287 innerTail11_51 best.bound = innerCandidates11_50 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_50 : scan edge (search edge 3) 4 [43] 43 innerTail11_50 best = best := by
  rw [innerTail11_50, scan, if_neg (by decide), innerCandidates11_50_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_51
    (if 3 ≤ innerCandidates11_50.length then search edge 3 [43, 8287] innerCandidates11_50 8330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_51

def innerTail11_49 : List Nat := 7753 :: innerTail11_50
def innerCandidates11_49 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_49_checked : candidates edge 4 43 7753 innerTail11_50 best.bound = innerCandidates11_49 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_49 : scan edge (search edge 3) 4 [43] 43 innerTail11_49 best = best := by
  rw [innerTail11_49, scan, if_neg (by decide), innerCandidates11_49_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_50
    (if 3 ≤ innerCandidates11_49.length then search edge 3 [43, 7753] innerCandidates11_49 7796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_50

def innerTail11_48 : List Nat := 7351 :: innerTail11_49
def innerCandidates11_48 : List Nat := [8287]
set_option maxRecDepth 4096 in
theorem innerCandidates11_48_checked : candidates edge 4 43 7351 innerTail11_49 best.bound = innerCandidates11_48 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_48 : scan edge (search edge 3) 4 [43] 43 innerTail11_48 best = best := by
  rw [innerTail11_48, scan, if_neg (by decide), innerCandidates11_48_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_49
    (if 3 ≤ innerCandidates11_48.length then search edge 3 [43, 7351] innerCandidates11_48 7394 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_49

def innerTail11_47 : List Nat := 7159 :: innerTail11_48
def innerCandidates11_47 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_47_checked : candidates edge 4 43 7159 innerTail11_48 best.bound = innerCandidates11_47 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_47 : scan edge (search edge 3) 4 [43] 43 innerTail11_47 best = best := by
  rw [innerTail11_47, scan, if_neg (by decide), innerCandidates11_47_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_48
    (if 3 ≤ innerCandidates11_47.length then search edge 3 [43, 7159] innerCandidates11_47 7202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_48

def innerTail11_46 : List Nat := 6967 :: innerTail11_47
def innerCandidates11_46 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_46_checked : candidates edge 4 43 6967 innerTail11_47 best.bound = innerCandidates11_46 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_46 : scan edge (search edge 3) 4 [43] 43 innerTail11_46 best = best := by
  rw [innerTail11_46, scan, if_neg (by decide), innerCandidates11_46_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_47
    (if 3 ≤ innerCandidates11_46.length then search edge 3 [43, 6967] innerCandidates11_46 7010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_47

def innerTail11_45 : List Nat := 6547 :: innerTail11_46
def innerCandidates11_45 : List Nat := [6967]
set_option maxRecDepth 4096 in
theorem innerCandidates11_45_checked : candidates edge 4 43 6547 innerTail11_46 best.bound = innerCandidates11_45 := by
  unfold candidates
  have row : edge 6547 = SearchData.row6547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_45 : scan edge (search edge 3) 4 [43] 43 innerTail11_45 best = best := by
  rw [innerTail11_45, scan, if_neg (by decide), innerCandidates11_45_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_46
    (if 3 ≤ innerCandidates11_45.length then search edge 3 [43, 6547] innerCandidates11_45 6590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_46

def innerTail11_44 : List Nat := 6529 :: innerTail11_45
def innerCandidates11_44 : List Nat := [6547, 8329]
set_option maxRecDepth 4096 in
theorem innerCandidates11_44_checked : candidates edge 4 43 6529 innerTail11_45 best.bound = innerCandidates11_44 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_44 : scan edge (search edge 3) 4 [43] 43 innerTail11_44 best = best := by
  rw [innerTail11_44, scan, if_neg (by decide), innerCandidates11_44_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_45
    (if 3 ≤ innerCandidates11_44.length then search edge 3 [43, 6529] innerCandidates11_44 6572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_45

def innerTail11_43 : List Nat := 6343 :: innerTail11_44
def innerCandidates11_43 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_43_checked : candidates edge 4 43 6343 innerTail11_44 best.bound = innerCandidates11_43 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_43 : scan edge (search edge 3) 4 [43] 43 innerTail11_43 best = best := by
  rw [innerTail11_43, scan, if_neg (by decide), innerCandidates11_43_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_44
    (if 3 ≤ innerCandidates11_43.length then search edge 3 [43, 6343] innerCandidates11_43 6386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_44

def innerTail11_42 : List Nat := 6091 :: innerTail11_43
def innerCandidates11_42 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_42_checked : candidates edge 4 43 6091 innerTail11_43 best.bound = innerCandidates11_42 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_42 : scan edge (search edge 3) 4 [43] 43 innerTail11_42 best = best := by
  rw [innerTail11_42, scan, if_neg (by decide), innerCandidates11_42_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_43
    (if 3 ≤ innerCandidates11_42.length then search edge 3 [43, 6091] innerCandidates11_42 6134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_43

def innerTail11_41 : List Nat := 5923 :: innerTail11_42
def innerCandidates11_41 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_41_checked : candidates edge 4 43 5923 innerTail11_42 best.bound = innerCandidates11_41 := by
  unfold candidates
  have row : edge 5923 = SearchData.row5923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_41 : scan edge (search edge 3) 4 [43] 43 innerTail11_41 best = best := by
  rw [innerTail11_41, scan, if_neg (by decide), innerCandidates11_41_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_42
    (if 3 ≤ innerCandidates11_41.length then search edge 3 [43, 5923] innerCandidates11_41 5966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_42

def innerTail11_40 : List Nat := 5857 :: innerTail11_41
def innerCandidates11_40 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_40_checked : candidates edge 4 43 5857 innerTail11_41 best.bound = innerCandidates11_40 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_40 : scan edge (search edge 3) 4 [43] 43 innerTail11_40 best = best := by
  rw [innerTail11_40, scan, if_neg (by decide), innerCandidates11_40_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_41
    (if 3 ≤ innerCandidates11_40.length then search edge 3 [43, 5857] innerCandidates11_40 5900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_41

def innerTail11_39 : List Nat := 5653 :: innerTail11_40
def innerCandidates11_39 : List Nat := [5923]
set_option maxRecDepth 4096 in
theorem innerCandidates11_39_checked : candidates edge 4 43 5653 innerTail11_40 best.bound = innerCandidates11_39 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_39 : scan edge (search edge 3) 4 [43] 43 innerTail11_39 best = best := by
  rw [innerTail11_39, scan, if_neg (by decide), innerCandidates11_39_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_40
    (if 3 ≤ innerCandidates11_39.length then search edge 3 [43, 5653] innerCandidates11_39 5696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_40

def innerTail11_38 : List Nat := 5569 :: innerTail11_39
def innerCandidates11_38 : List Nat := [5923]
set_option maxRecDepth 4096 in
theorem innerCandidates11_38_checked : candidates edge 4 43 5569 innerTail11_39 best.bound = innerCandidates11_38 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_38 : scan edge (search edge 3) 4 [43] 43 innerTail11_38 best = best := by
  rw [innerTail11_38, scan, if_neg (by decide), innerCandidates11_38_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_39
    (if 3 ≤ innerCandidates11_38.length then search edge 3 [43, 5569] innerCandidates11_38 5612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_39

def innerTail11_37 : List Nat := 5563 :: innerTail11_38
def innerCandidates11_37 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_37_checked : candidates edge 4 43 5563 innerTail11_38 best.bound = innerCandidates11_37 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_37 : scan edge (search edge 3) 4 [43] 43 innerTail11_37 best = best := by
  rw [innerTail11_37, scan, if_neg (by decide), innerCandidates11_37_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_38
    (if 3 ≤ innerCandidates11_37.length then search edge 3 [43, 5563] innerCandidates11_37 5606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_38

def innerTail11_36 : List Nat := 4813 :: innerTail11_37
def innerCandidates11_36 : List Nat := [7351]
set_option maxRecDepth 4096 in
theorem innerCandidates11_36_checked : candidates edge 4 43 4813 innerTail11_37 best.bound = innerCandidates11_36 := by
  unfold candidates
  have row : edge 4813 = SearchData.row4813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_36 : scan edge (search edge 3) 4 [43] 43 innerTail11_36 best = best := by
  rw [innerTail11_36, scan, if_neg (by decide), innerCandidates11_36_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_37
    (if 3 ≤ innerCandidates11_36.length then search edge 3 [43, 4813] innerCandidates11_36 4856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_37

def innerTail11_35 : List Nat := 4783 :: innerTail11_36
def innerCandidates11_35 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_35_checked : candidates edge 4 43 4783 innerTail11_36 best.bound = innerCandidates11_35 := by
  unfold candidates
  have row : edge 4783 = SearchData.row4783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_35 : scan edge (search edge 3) 4 [43] 43 innerTail11_35 best = best := by
  rw [innerTail11_35, scan, if_neg (by decide), innerCandidates11_35_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_36
    (if 3 ≤ innerCandidates11_35.length then search edge 3 [43, 4783] innerCandidates11_35 4826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_36

def innerTail11_34 : List Nat := 4363 :: innerTail11_35
def innerCandidates11_34 : List Nat := [4813, 5653]
set_option maxRecDepth 4096 in
theorem innerCandidates11_34_checked : candidates edge 4 43 4363 innerTail11_35 best.bound = innerCandidates11_34 := by
  unfold candidates
  have row : edge 4363 = SearchData.row4363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_34 : scan edge (search edge 3) 4 [43] 43 innerTail11_34 best = best := by
  rw [innerTail11_34, scan, if_neg (by decide), innerCandidates11_34_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_35
    (if 3 ≤ innerCandidates11_34.length then search edge 3 [43, 4363] innerCandidates11_34 4406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_35

def innerTail11_33 : List Nat := 4243 :: innerTail11_34
def innerCandidates11_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_33_checked : candidates edge 4 43 4243 innerTail11_34 best.bound = innerCandidates11_33 := by
  unfold candidates
  have row : edge 4243 = SearchData.row4243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_33 : scan edge (search edge 3) 4 [43] 43 innerTail11_33 best = best := by
  rw [innerTail11_33, scan, if_neg (by decide), innerCandidates11_33_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_34
    (if 3 ≤ innerCandidates11_33.length then search edge 3 [43, 4243] innerCandidates11_33 4286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_34

def innerTail11_32 : List Nat := 4111 :: innerTail11_33
def innerCandidates11_32 : List Nat := [5857]
set_option maxRecDepth 4096 in
theorem innerCandidates11_32_checked : candidates edge 4 43 4111 innerTail11_33 best.bound = innerCandidates11_32 := by
  unfold candidates
  have row : edge 4111 = SearchData.row4111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_32 : scan edge (search edge 3) 4 [43] 43 innerTail11_32 best = best := by
  rw [innerTail11_32, scan, if_neg (by decide), innerCandidates11_32_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_33
    (if 3 ≤ innerCandidates11_32.length then search edge 3 [43, 4111] innerCandidates11_32 4154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_33

def innerTail11_31 : List Nat := 3931 :: innerTail11_32
def innerCandidates11_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_31_checked : candidates edge 4 43 3931 innerTail11_32 best.bound = innerCandidates11_31 := by
  unfold candidates
  have row : edge 3931 = SearchData.row3931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_31 : scan edge (search edge 3) 4 [43] 43 innerTail11_31 best = best := by
  rw [innerTail11_31, scan, if_neg (by decide), innerCandidates11_31_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_32
    (if 3 ≤ innerCandidates11_31.length then search edge 3 [43, 3931] innerCandidates11_31 3974 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_32

def innerTail11_30 : List Nat := 3877 :: innerTail11_31
def innerCandidates11_30 : List Nat := [7159]
set_option maxRecDepth 4096 in
theorem innerCandidates11_30_checked : candidates edge 4 43 3877 innerTail11_31 best.bound = innerCandidates11_30 := by
  unfold candidates
  have row : edge 3877 = SearchData.row3877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_30 : scan edge (search edge 3) 4 [43] 43 innerTail11_30 best = best := by
  rw [innerTail11_30, scan, if_neg (by decide), innerCandidates11_30_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_31
    (if 3 ≤ innerCandidates11_30.length then search edge 3 [43, 3877] innerCandidates11_30 3920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_31

def innerTail11_29 : List Nat := 3469 :: innerTail11_30
def innerCandidates11_29 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_29_checked : candidates edge 4 43 3469 innerTail11_30 best.bound = innerCandidates11_29 := by
  unfold candidates
  have row : edge 3469 = SearchData.row3469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_29 : scan edge (search edge 3) 4 [43] 43 innerTail11_29 best = best := by
  rw [innerTail11_29, scan, if_neg (by decide), innerCandidates11_29_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_30
    (if 3 ≤ innerCandidates11_29.length then search edge 3 [43, 3469] innerCandidates11_29 3512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_30

def innerTail11_28 : List Nat := 3373 :: innerTail11_29
def innerCandidates11_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_28_checked : candidates edge 4 43 3373 innerTail11_29 best.bound = innerCandidates11_28 := by
  unfold candidates
  have row : edge 3373 = SearchData.row3373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_28 : scan edge (search edge 3) 4 [43] 43 innerTail11_28 best = best := by
  rw [innerTail11_28, scan, if_neg (by decide), innerCandidates11_28_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_29
    (if 3 ≤ innerCandidates11_28.length then search edge 3 [43, 3373] innerCandidates11_28 3416 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_29

def innerTail11_27 : List Nat := 3361 :: innerTail11_28
def innerCandidates11_27 : List Nat := [6343]
set_option maxRecDepth 4096 in
theorem innerCandidates11_27_checked : candidates edge 4 43 3361 innerTail11_28 best.bound = innerCandidates11_27 := by
  unfold candidates
  have row : edge 3361 = SearchData.row3361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_27 : scan edge (search edge 3) 4 [43] 43 innerTail11_27 best = best := by
  rw [innerTail11_27, scan, if_neg (by decide), innerCandidates11_27_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_28
    (if 3 ≤ innerCandidates11_27.length then search edge 3 [43, 3361] innerCandidates11_27 3404 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_28

def innerTail11_26 : List Nat := 3319 :: innerTail11_27
def innerCandidates11_26 : List Nat := [3373, 4363]
set_option maxRecDepth 4096 in
theorem innerCandidates11_26_checked : candidates edge 4 43 3319 innerTail11_27 best.bound = innerCandidates11_26 := by
  unfold candidates
  have row : edge 3319 = SearchData.row3319.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_26 : scan edge (search edge 3) 4 [43] 43 innerTail11_26 best = best := by
  rw [innerTail11_26, scan, if_neg (by decide), innerCandidates11_26_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_27
    (if 3 ≤ innerCandidates11_26.length then search edge 3 [43, 3319] innerCandidates11_26 3362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_27

def innerTail11_25 : List Nat := 3259 :: innerTail11_26
def innerCandidates11_25 : List Nat := [5569]
set_option maxRecDepth 4096 in
theorem innerCandidates11_25_checked : candidates edge 4 43 3259 innerTail11_26 best.bound = innerCandidates11_25 := by
  unfold candidates
  have row : edge 3259 = SearchData.row3259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_25 : scan edge (search edge 3) 4 [43] 43 innerTail11_25 best = best := by
  rw [innerTail11_25, scan, if_neg (by decide), innerCandidates11_25_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_26
    (if 3 ≤ innerCandidates11_25.length then search edge 3 [43, 3259] innerCandidates11_25 3302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_26

def innerTail11_24 : List Nat := 3253 :: innerTail11_25
def innerCandidates11_24 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_24_checked : candidates edge 4 43 3253 innerTail11_25 best.bound = innerCandidates11_24 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_24 : scan edge (search edge 3) 4 [43] 43 innerTail11_24 best = best := by
  rw [innerTail11_24, scan, if_neg (by decide), innerCandidates11_24_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_25
    (if 3 ≤ innerCandidates11_24.length then search edge 3 [43, 3253] innerCandidates11_24 3296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_25

def innerTail11_23 : List Nat := 3187 :: innerTail11_24
def innerCandidates11_23 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_23_checked : candidates edge 4 43 3187 innerTail11_24 best.bound = innerCandidates11_23 := by
  unfold candidates
  have row : edge 3187 = SearchData.row3187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_23 : scan edge (search edge 3) 4 [43] 43 innerTail11_23 best = best := by
  rw [innerTail11_23, scan, if_neg (by decide), innerCandidates11_23_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_24
    (if 3 ≤ innerCandidates11_23.length then search edge 3 [43, 3187] innerCandidates11_23 3230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_24

def innerTail11_22 : List Nat := 3049 :: innerTail11_23
def innerCandidates11_22 : List Nat := [3259]
set_option maxRecDepth 4096 in
theorem innerCandidates11_22_checked : candidates edge 4 43 3049 innerTail11_23 best.bound = innerCandidates11_22 := by
  unfold candidates
  have row : edge 3049 = SearchData.row3049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_22 : scan edge (search edge 3) 4 [43] 43 innerTail11_22 best = best := by
  rw [innerTail11_22, scan, if_neg (by decide), innerCandidates11_22_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_23
    (if 3 ≤ innerCandidates11_22.length then search edge 3 [43, 3049] innerCandidates11_22 3092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_23

def innerTail11_21 : List Nat := 2749 :: innerTail11_22
def innerCandidates11_21 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates11_21_checked : candidates edge 4 43 2749 innerTail11_22 best.bound = innerCandidates11_21 := by
  unfold candidates
  have row : edge 2749 = SearchData.row2749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_21 : scan edge (search edge 3) 4 [43] 43 innerTail11_21 best = best := by
  rw [innerTail11_21, scan, if_neg (by decide), innerCandidates11_21_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_22
    (if 3 ≤ innerCandidates11_21.length then search edge 3 [43, 2749] innerCandidates11_21 2792 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_22

def innerTail11_20 : List Nat := 2557 :: innerTail11_21
def innerCandidates11_20 : List Nat := [4243]
set_option maxRecDepth 4096 in
theorem innerCandidates11_20_checked : candidates edge 4 43 2557 innerTail11_21 best.bound = innerCandidates11_20 := by
  unfold candidates
  have row : edge 2557 = PrimePairCache.smallRow2557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_20 : scan edge (search edge 3) 4 [43] 43 innerTail11_20 best = best := by
  rw [innerTail11_20, scan, if_neg (by decide), innerCandidates11_20_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_21
    (if 3 ≤ innerCandidates11_20.length then search edge 3 [43, 2557] innerCandidates11_20 2600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_21

def innerTail11_19 : List Nat := 2503 :: innerTail11_20
def innerCandidates11_19 : List Nat := [6967]
set_option maxRecDepth 4096 in
theorem innerCandidates11_19_checked : candidates edge 4 43 2503 innerTail11_20 best.bound = innerCandidates11_19 := by
  unfold candidates
  have row : edge 2503 = PrimePairCache.smallRow2503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_19 : scan edge (search edge 3) 4 [43] 43 innerTail11_19 best = best := by
  rw [innerTail11_19, scan, if_neg (by decide), innerCandidates11_19_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_20
    (if 3 ≤ innerCandidates11_19.length then search edge 3 [43, 2503] innerCandidates11_19 2546 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_20

def innerTail11_18 : List Nat := 2389 :: innerTail11_19
def innerCandidates11_18 : List Nat := [2503, 4243, 6529, 7351]
set_option maxRecDepth 4096 in
theorem innerCandidates11_18_checked : candidates edge 4 43 2389 innerTail11_19 best.bound = innerCandidates11_18 := by
  unfold candidates
  have row : edge 2389 = PrimePairCache.smallRow2389.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild11_18 : search edge 3 [43, 2389] innerCandidates11_18 2432 best = best := by decide

theorem innerScan11_18 : scan edge (search edge 3) 4 [43] 43 innerTail11_18 best = best := by
  rw [innerTail11_18, scan, if_neg (by decide), innerCandidates11_18_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_19
    (if 3 ≤ innerCandidates11_18.length then search edge 3 [43, 2389] innerCandidates11_18 2432 best else best) = best
  rw [if_pos (by decide), innerChild11_18]
  exact innerScan11_19

def innerTail11_17 : List Nat := 2251 :: innerTail11_18
def innerCandidates11_17 : List Nat := [4111]
set_option maxRecDepth 4096 in
theorem innerCandidates11_17_checked : candidates edge 4 43 2251 innerTail11_18 best.bound = innerCandidates11_17 := by
  unfold candidates
  have row : edge 2251 = PrimePairCache.smallRow2251.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_17 : scan edge (search edge 3) 4 [43] 43 innerTail11_17 best = best := by
  rw [innerTail11_17, scan, if_neg (by decide), innerCandidates11_17_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_18
    (if 3 ≤ innerCandidates11_17.length then search edge 3 [43, 2251] innerCandidates11_17 2294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_18

def innerTail11_16 : List Nat := 1993 :: innerTail11_17
def innerCandidates11_16 : List Nat := [8287]
set_option maxRecDepth 4096 in
theorem innerCandidates11_16_checked : candidates edge 4 43 1993 innerTail11_17 best.bound = innerCandidates11_16 := by
  unfold candidates
  have row : edge 1993 = PrimePairCache.smallRow1993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_16 : scan edge (search edge 3) 4 [43] 43 innerTail11_16 best = best := by
  rw [innerTail11_16, scan, if_neg (by decide), innerCandidates11_16_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_17
    (if 3 ≤ innerCandidates11_16.length then search edge 3 [43, 1993] innerCandidates11_16 2036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_17

def innerTail11_15 : List Nat := 1867 :: innerTail11_16
def innerCandidates11_15 : List Nat := [4111]
set_option maxRecDepth 4096 in
theorem innerCandidates11_15_checked : candidates edge 4 43 1867 innerTail11_16 best.bound = innerCandidates11_15 := by
  unfold candidates
  have row : edge 1867 = PrimePairCache.smallRow1867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_15 : scan edge (search edge 3) 4 [43] 43 innerTail11_15 best = best := by
  rw [innerTail11_15, scan, if_neg (by decide), innerCandidates11_15_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_16
    (if 3 ≤ innerCandidates11_15.length then search edge 3 [43, 1867] innerCandidates11_15 1910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_16

def innerTail11_14 : List Nat := 1777 :: innerTail11_15
def innerCandidates11_14 : List Nat := [3253]
set_option maxRecDepth 4096 in
theorem innerCandidates11_14_checked : candidates edge 4 43 1777 innerTail11_15 best.bound = innerCandidates11_14 := by
  unfold candidates
  have row : edge 1777 = PrimePairCache.smallRow1777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_14 : scan edge (search edge 3) 4 [43] 43 innerTail11_14 best = best := by
  rw [innerTail11_14, scan, if_neg (by decide), innerCandidates11_14_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_15
    (if 3 ≤ innerCandidates11_14.length then search edge 3 [43, 1777] innerCandidates11_14 1820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_15

def innerTail11_13 : List Nat := 1693 :: innerTail11_14
def innerCandidates11_13 : List Nat := [3373]
set_option maxRecDepth 4096 in
theorem innerCandidates11_13_checked : candidates edge 4 43 1693 innerTail11_14 best.bound = innerCandidates11_13 := by
  unfold candidates
  have row : edge 1693 = PrimePairCache.smallRow1693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_13 : scan edge (search edge 3) 4 [43] 43 innerTail11_13 best = best := by
  rw [innerTail11_13, scan, if_neg (by decide), innerCandidates11_13_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_14
    (if 3 ≤ innerCandidates11_13.length then search edge 3 [43, 1693] innerCandidates11_13 1736 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_14

def innerTail11_12 : List Nat := 1621 :: innerTail11_13
def innerCandidates11_12 : List Nat := [3187]
set_option maxRecDepth 4096 in
theorem innerCandidates11_12_checked : candidates edge 4 43 1621 innerTail11_13 best.bound = innerCandidates11_12 := by
  unfold candidates
  have row : edge 1621 = PrimePairCache.smallRow1621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_12 : scan edge (search edge 3) 4 [43] 43 innerTail11_12 best = best := by
  rw [innerTail11_12, scan, if_neg (by decide), innerCandidates11_12_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_13
    (if 3 ≤ innerCandidates11_12.length then search edge 3 [43, 1621] innerCandidates11_12 1664 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_13

def innerTail11_11 : List Nat := 1399 :: innerTail11_12
def innerCandidates11_11 : List Nat := [6529]
set_option maxRecDepth 4096 in
theorem innerCandidates11_11_checked : candidates edge 4 43 1399 innerTail11_12 best.bound = innerCandidates11_11 := by
  unfold candidates
  have row : edge 1399 = PrimePairCache.smallRow1399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_11 : scan edge (search edge 3) 4 [43] 43 innerTail11_11 best = best := by
  rw [innerTail11_11, scan, if_neg (by decide), innerCandidates11_11_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_12
    (if 3 ≤ innerCandidates11_11.length then search edge 3 [43, 1399] innerCandidates11_11 1442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_12

def innerTail11_10 : List Nat := 1381 :: innerTail11_11
def innerCandidates11_10 : List Nat := [4111]
set_option maxRecDepth 4096 in
theorem innerCandidates11_10_checked : candidates edge 4 43 1381 innerTail11_11 best.bound = innerCandidates11_10 := by
  unfold candidates
  have row : edge 1381 = PrimePairCache.smallRow1381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_10 : scan edge (search edge 3) 4 [43] 43 innerTail11_10 best = best := by
  rw [innerTail11_10, scan, if_neg (by decide), innerCandidates11_10_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_11
    (if 3 ≤ innerCandidates11_10.length then search edge 3 [43, 1381] innerCandidates11_10 1424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_11

def innerTail11_9 : List Nat := 1213 :: innerTail11_10
def innerCandidates11_9 : List Nat := [1621, 6547]
set_option maxRecDepth 4096 in
theorem innerCandidates11_9_checked : candidates edge 4 43 1213 innerTail11_10 best.bound = innerCandidates11_9 := by
  unfold candidates
  have row : edge 1213 = PrimePairCache.smallRow1213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_9 : scan edge (search edge 3) 4 [43] 43 innerTail11_9 best = best := by
  rw [innerTail11_9, scan, if_neg (by decide), innerCandidates11_9_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_10
    (if 3 ≤ innerCandidates11_9.length then search edge 3 [43, 1213] innerCandidates11_9 1256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_10

def innerTail11_8 : List Nat := 1153 :: innerTail11_9
def innerCandidates11_8 : List Nat := [3253]
set_option maxRecDepth 4096 in
theorem innerCandidates11_8_checked : candidates edge 4 43 1153 innerTail11_9 best.bound = innerCandidates11_8 := by
  unfold candidates
  have row : edge 1153 = PrimePairCache.smallRow1153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_8 : scan edge (search edge 3) 4 [43] 43 innerTail11_8 best = best := by
  rw [innerTail11_8, scan, if_neg (by decide), innerCandidates11_8_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_9
    (if 3 ≤ innerCandidates11_8.length then search edge 3 [43, 1153] innerCandidates11_8 1196 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_9

def innerTail11_7 : List Nat := 1051 :: innerTail11_8
def innerCandidates11_7 : List Nat := [2389, 4813]
set_option maxRecDepth 4096 in
theorem innerCandidates11_7_checked : candidates edge 4 43 1051 innerTail11_8 best.bound = innerCandidates11_7 := by
  unfold candidates
  have row : edge 1051 = PrimePairCache.smallRow1051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_7 : scan edge (search edge 3) 4 [43] 43 innerTail11_7 best = best := by
  rw [innerTail11_7, scan, if_neg (by decide), innerCandidates11_7_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_8
    (if 3 ≤ innerCandidates11_7.length then search edge 3 [43, 1051] innerCandidates11_7 1094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_8

def innerTail11_6 : List Nat := 691 :: innerTail11_7
def innerCandidates11_6 : List Nat := [1693, 3469]
set_option maxRecDepth 4096 in
theorem innerCandidates11_6_checked : candidates edge 4 43 691 innerTail11_7 best.bound = innerCandidates11_6 := by
  unfold candidates
  have row : edge 691 = PrimePairCache.smallRow691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_6 : scan edge (search edge 3) 4 [43] 43 innerTail11_6 best = best := by
  rw [innerTail11_6, scan, if_neg (by decide), innerCandidates11_6_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_7
    (if 3 ≤ innerCandidates11_6.length then search edge 3 [43, 691] innerCandidates11_6 734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_7

def innerTail11_5 : List Nat := 613 :: innerTail11_6
def innerCandidates11_5 : List Nat := [4363, 6547, 6967]
set_option maxRecDepth 4096 in
theorem innerCandidates11_5_checked : candidates edge 4 43 613 innerTail11_6 best.bound = innerCandidates11_5 := by
  unfold candidates
  have row : edge 613 = PrimePairCache.smallRow613.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild11_5 : search edge 3 [43, 613] innerCandidates11_5 656 best = best := by decide

theorem innerScan11_5 : scan edge (search edge 3) 4 [43] 43 innerTail11_5 best = best := by
  rw [innerTail11_5, scan, if_neg (by decide), innerCandidates11_5_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_6
    (if 3 ≤ innerCandidates11_5.length then search edge 3 [43, 613] innerCandidates11_5 656 best else best) = best
  rw [if_pos (by decide), innerChild11_5]
  exact innerScan11_6

def innerTail11_4 : List Nat := 499 :: innerTail11_5
def innerCandidates11_4 : List Nat := [691]
set_option maxRecDepth 4096 in
theorem innerCandidates11_4_checked : candidates edge 4 43 499 innerTail11_5 best.bound = innerCandidates11_4 := by
  unfold candidates
  have row : edge 499 = PrimePairCache.smallRow499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_4 : scan edge (search edge 3) 4 [43] 43 innerTail11_4 best = best := by
  rw [innerTail11_4, scan, if_neg (by decide), innerCandidates11_4_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_5
    (if 3 ≤ innerCandidates11_4.length then search edge 3 [43, 499] innerCandidates11_4 542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_5

def innerTail11_3 : List Nat := 271 :: innerTail11_4
def innerCandidates11_3 : List Nat := [1213, 1399, 4111]
set_option maxRecDepth 4096 in
theorem innerCandidates11_3_checked : candidates edge 4 43 271 innerTail11_4 best.bound = innerCandidates11_3 := by
  unfold candidates
  have row : edge 271 = PrimePairCache.smallRow271.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild11_3 : search edge 3 [43, 271] innerCandidates11_3 314 best = best := by decide

theorem innerScan11_3 : scan edge (search edge 3) 4 [43] 43 innerTail11_3 best = best := by
  rw [innerTail11_3, scan, if_neg (by decide), innerCandidates11_3_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_4
    (if 3 ≤ innerCandidates11_3.length then search edge 3 [43, 271] innerCandidates11_3 314 best else best) = best
  rw [if_pos (by decide), innerChild11_3]
  exact innerScan11_4

def innerTail11_2 : List Nat := 223 :: innerTail11_3
def innerCandidates11_2 : List Nat := [7159]
set_option maxRecDepth 4096 in
theorem innerCandidates11_2_checked : candidates edge 4 43 223 innerTail11_3 best.bound = innerCandidates11_2 := by
  unfold candidates
  have row : edge 223 = PrimePairCache.smallRow223.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_2 : scan edge (search edge 3) 4 [43] 43 innerTail11_2 best = best := by
  rw [innerTail11_2, scan, if_neg (by decide), innerCandidates11_2_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_3
    (if 3 ≤ innerCandidates11_2.length then search edge 3 [43, 223] innerCandidates11_2 266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_3

def innerTail11_1 : List Nat := 103 :: innerTail11_2
def innerCandidates11_1 : List Nat := [8329]
set_option maxRecDepth 4096 in
theorem innerCandidates11_1_checked : candidates edge 4 43 103 innerTail11_2 best.bound = innerCandidates11_1 := by
  unfold candidates
  have row : edge 103 = PrimePairCache.smallRow103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan11_1 : scan edge (search edge 3) 4 [43] 43 innerTail11_1 best = best := by
  rw [innerTail11_1, scan, if_neg (by decide), innerCandidates11_1_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_2
    (if 3 ≤ innerCandidates11_1.length then search edge 3 [43, 103] innerCandidates11_1 146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan11_2

def innerTail11_0 : List Nat := 97 :: innerTail11_1
def innerCandidates11_0 : List Nat := [1381, 3373, 8287]
set_option maxRecDepth 4096 in
theorem innerCandidates11_0_checked : candidates edge 4 43 97 innerTail11_1 best.bound = innerCandidates11_0 := by
  unfold candidates
  have row : edge 97 = PrimePairCache.smallRow97.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild11_0 : search edge 3 [43, 97] innerCandidates11_0 140 best = best := by decide

theorem innerScan11_0 : scan edge (search edge 3) 4 [43] 43 innerTail11_0 best = best := by
  rw [innerTail11_0, scan, if_neg (by decide), innerCandidates11_0_checked]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_1
    (if 3 ≤ innerCandidates11_0.length then search edge 3 [43, 97] innerCandidates11_0 140 best else best) = best
  rw [if_pos (by decide), innerChild11_0]
  exact innerScan11_1

theorem rootBranch11 : search edge 4 [43] rootCandidates11 43 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [43] 43 innerTail11_0 best = best
  exact innerScan11_0

end ProjectEulerSolutions.PrimePairUpper.Search
