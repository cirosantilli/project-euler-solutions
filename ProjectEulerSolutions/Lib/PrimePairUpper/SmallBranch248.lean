import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates19 : List Nat := [193, 241, 367, 397, 613, 631, 967, 1093, 1117, 1447, 1627, 1657, 1801, 1933, 1993, 2179, 2371, 2551, 2593, 2647, 2689, 2797, 3181, 3253, 3343, 3511, 3517, 3547, 3607, 3733, 3853, 4153, 4231, 4483, 4831, 4999, 5077, 5101, 5323, 5449, 5503, 5581, 5647, 5653, 5659, 5791, 6091, 6217, 6373, 6451, 6553, 6673, 6781, 6871, 7057, 7207, 7549, 7561, 7591, 7933, 8179, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates19_checked : candidates edge 5 0 79 tail20 best.bound = rootCandidates19 := by
  unfold candidates
  have row : edge 79 = PrimePairCache.smallRow79.contains := by funext q; rfl
  rw [row]
  decide

def innerTail19_62 : List Nat := []
theorem innerScan19_62 : scan edge (search edge 3) 4 [79] 79 innerTail19_62 best = best := rfl

def innerTail19_61 : List Nat := 8263 :: innerTail19_62
def innerCandidates19_61 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_61_checked : candidates edge 4 79 8263 innerTail19_62 best.bound = innerCandidates19_61 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_61 : scan edge (search edge 3) 4 [79] 79 innerTail19_61 best = best := by
  rw [innerTail19_61, scan, if_neg (by decide), innerCandidates19_61_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_62
    (if 3 ≤ innerCandidates19_61.length then search edge 3 [79, 8263] innerCandidates19_61 8342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_62

def innerTail19_60 : List Nat := 8179 :: innerTail19_61
def innerCandidates19_60 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_60_checked : candidates edge 4 79 8179 innerTail19_61 best.bound = innerCandidates19_60 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_60 : scan edge (search edge 3) 4 [79] 79 innerTail19_60 best = best := by
  rw [innerTail19_60, scan, if_neg (by decide), innerCandidates19_60_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_61
    (if 3 ≤ innerCandidates19_60.length then search edge 3 [79, 8179] innerCandidates19_60 8258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_61

def innerTail19_59 : List Nat := 7933 :: innerTail19_60
def innerCandidates19_59 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_59_checked : candidates edge 4 79 7933 innerTail19_60 best.bound = innerCandidates19_59 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_59 : scan edge (search edge 3) 4 [79] 79 innerTail19_59 best = best := by
  rw [innerTail19_59, scan, if_neg (by decide), innerCandidates19_59_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_60
    (if 3 ≤ innerCandidates19_59.length then search edge 3 [79, 7933] innerCandidates19_59 8012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_60

def innerTail19_58 : List Nat := 7591 :: innerTail19_59
def innerCandidates19_58 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_58_checked : candidates edge 4 79 7591 innerTail19_59 best.bound = innerCandidates19_58 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_58 : scan edge (search edge 3) 4 [79] 79 innerTail19_58 best = best := by
  rw [innerTail19_58, scan, if_neg (by decide), innerCandidates19_58_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_59
    (if 3 ≤ innerCandidates19_58.length then search edge 3 [79, 7591] innerCandidates19_58 7670 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_59

def innerTail19_57 : List Nat := 7561 :: innerTail19_58
def innerCandidates19_57 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_57_checked : candidates edge 4 79 7561 innerTail19_58 best.bound = innerCandidates19_57 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_57 : scan edge (search edge 3) 4 [79] 79 innerTail19_57 best = best := by
  rw [innerTail19_57, scan, if_neg (by decide), innerCandidates19_57_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_58
    (if 3 ≤ innerCandidates19_57.length then search edge 3 [79, 7561] innerCandidates19_57 7640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_58

def innerTail19_56 : List Nat := 7549 :: innerTail19_57
def innerCandidates19_56 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_56_checked : candidates edge 4 79 7549 innerTail19_57 best.bound = innerCandidates19_56 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_56 : scan edge (search edge 3) 4 [79] 79 innerTail19_56 best = best := by
  rw [innerTail19_56, scan, if_neg (by decide), innerCandidates19_56_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_57
    (if 3 ≤ innerCandidates19_56.length then search edge 3 [79, 7549] innerCandidates19_56 7628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_57

def innerTail19_55 : List Nat := 7207 :: innerTail19_56
def innerCandidates19_55 : List Nat := [8179]
set_option maxRecDepth 4096 in
theorem innerCandidates19_55_checked : candidates edge 4 79 7207 innerTail19_56 best.bound = innerCandidates19_55 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_55 : scan edge (search edge 3) 4 [79] 79 innerTail19_55 best = best := by
  rw [innerTail19_55, scan, if_neg (by decide), innerCandidates19_55_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_56
    (if 3 ≤ innerCandidates19_55.length then search edge 3 [79, 7207] innerCandidates19_55 7286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_56

def innerTail19_54 : List Nat := 7057 :: innerTail19_55
def innerCandidates19_54 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_54_checked : candidates edge 4 79 7057 innerTail19_55 best.bound = innerCandidates19_54 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_54 : scan edge (search edge 3) 4 [79] 79 innerTail19_54 best = best := by
  rw [innerTail19_54, scan, if_neg (by decide), innerCandidates19_54_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_55
    (if 3 ≤ innerCandidates19_54.length then search edge 3 [79, 7057] innerCandidates19_54 7136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_55

def innerTail19_53 : List Nat := 6871 :: innerTail19_54
def innerCandidates19_53 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_53_checked : candidates edge 4 79 6871 innerTail19_54 best.bound = innerCandidates19_53 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_53 : scan edge (search edge 3) 4 [79] 79 innerTail19_53 best = best := by
  rw [innerTail19_53, scan, if_neg (by decide), innerCandidates19_53_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_54
    (if 3 ≤ innerCandidates19_53.length then search edge 3 [79, 6871] innerCandidates19_53 6950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_54

def innerTail19_52 : List Nat := 6781 :: innerTail19_53
def innerCandidates19_52 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_52_checked : candidates edge 4 79 6781 innerTail19_53 best.bound = innerCandidates19_52 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_52 : scan edge (search edge 3) 4 [79] 79 innerTail19_52 best = best := by
  rw [innerTail19_52, scan, if_neg (by decide), innerCandidates19_52_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_53
    (if 3 ≤ innerCandidates19_52.length then search edge 3 [79, 6781] innerCandidates19_52 6860 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_53

def innerTail19_51 : List Nat := 6673 :: innerTail19_52
def innerCandidates19_51 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_51_checked : candidates edge 4 79 6673 innerTail19_52 best.bound = innerCandidates19_51 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_51 : scan edge (search edge 3) 4 [79] 79 innerTail19_51 best = best := by
  rw [innerTail19_51, scan, if_neg (by decide), innerCandidates19_51_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_52
    (if 3 ≤ innerCandidates19_51.length then search edge 3 [79, 6673] innerCandidates19_51 6752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_52

def innerTail19_50 : List Nat := 6553 :: innerTail19_51
def innerCandidates19_50 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_50_checked : candidates edge 4 79 6553 innerTail19_51 best.bound = innerCandidates19_50 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_50 : scan edge (search edge 3) 4 [79] 79 innerTail19_50 best = best := by
  rw [innerTail19_50, scan, if_neg (by decide), innerCandidates19_50_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_51
    (if 3 ≤ innerCandidates19_50.length then search edge 3 [79, 6553] innerCandidates19_50 6632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_51

def innerTail19_49 : List Nat := 6451 :: innerTail19_50
def innerCandidates19_49 : List Nat := [7561]
set_option maxRecDepth 4096 in
theorem innerCandidates19_49_checked : candidates edge 4 79 6451 innerTail19_50 best.bound = innerCandidates19_49 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_49 : scan edge (search edge 3) 4 [79] 79 innerTail19_49 best = best := by
  rw [innerTail19_49, scan, if_neg (by decide), innerCandidates19_49_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_50
    (if 3 ≤ innerCandidates19_49.length then search edge 3 [79, 6451] innerCandidates19_49 6530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_50

def innerTail19_48 : List Nat := 6373 :: innerTail19_49
def innerCandidates19_48 : List Nat := [6781]
set_option maxRecDepth 4096 in
theorem innerCandidates19_48_checked : candidates edge 4 79 6373 innerTail19_49 best.bound = innerCandidates19_48 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_48 : scan edge (search edge 3) 4 [79] 79 innerTail19_48 best = best := by
  rw [innerTail19_48, scan, if_neg (by decide), innerCandidates19_48_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_49
    (if 3 ≤ innerCandidates19_48.length then search edge 3 [79, 6373] innerCandidates19_48 6452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_49

def innerTail19_47 : List Nat := 6217 :: innerTail19_48
def innerCandidates19_47 : List Nat := [7933]
set_option maxRecDepth 4096 in
theorem innerCandidates19_47_checked : candidates edge 4 79 6217 innerTail19_48 best.bound = innerCandidates19_47 := by
  unfold candidates
  have row : edge 6217 = SearchData.row6217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_47 : scan edge (search edge 3) 4 [79] 79 innerTail19_47 best = best := by
  rw [innerTail19_47, scan, if_neg (by decide), innerCandidates19_47_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_48
    (if 3 ≤ innerCandidates19_47.length then search edge 3 [79, 6217] innerCandidates19_47 6296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_48

def innerTail19_46 : List Nat := 6091 :: innerTail19_47
def innerCandidates19_46 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_46_checked : candidates edge 4 79 6091 innerTail19_47 best.bound = innerCandidates19_46 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_46 : scan edge (search edge 3) 4 [79] 79 innerTail19_46 best = best := by
  rw [innerTail19_46, scan, if_neg (by decide), innerCandidates19_46_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_47
    (if 3 ≤ innerCandidates19_46.length then search edge 3 [79, 6091] innerCandidates19_46 6170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_47

def innerTail19_45 : List Nat := 5791 :: innerTail19_46
def innerCandidates19_45 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_45_checked : candidates edge 4 79 5791 innerTail19_46 best.bound = innerCandidates19_45 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_45 : scan edge (search edge 3) 4 [79] 79 innerTail19_45 best = best := by
  rw [innerTail19_45, scan, if_neg (by decide), innerCandidates19_45_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_46
    (if 3 ≤ innerCandidates19_45.length then search edge 3 [79, 5791] innerCandidates19_45 5870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_46

def innerTail19_44 : List Nat := 5659 :: innerTail19_45
def innerCandidates19_44 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_44_checked : candidates edge 4 79 5659 innerTail19_45 best.bound = innerCandidates19_44 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_44 : scan edge (search edge 3) 4 [79] 79 innerTail19_44 best = best := by
  rw [innerTail19_44, scan, if_neg (by decide), innerCandidates19_44_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_45
    (if 3 ≤ innerCandidates19_44.length then search edge 3 [79, 5659] innerCandidates19_44 5738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_45

def innerTail19_43 : List Nat := 5653 :: innerTail19_44
def innerCandidates19_43 : List Nat := [7549]
set_option maxRecDepth 4096 in
theorem innerCandidates19_43_checked : candidates edge 4 79 5653 innerTail19_44 best.bound = innerCandidates19_43 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_43 : scan edge (search edge 3) 4 [79] 79 innerTail19_43 best = best := by
  rw [innerTail19_43, scan, if_neg (by decide), innerCandidates19_43_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_44
    (if 3 ≤ innerCandidates19_43.length then search edge 3 [79, 5653] innerCandidates19_43 5732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_44

def innerTail19_42 : List Nat := 5647 :: innerTail19_43
def innerCandidates19_42 : List Nat := [5659, 6373]
set_option maxRecDepth 4096 in
theorem innerCandidates19_42_checked : candidates edge 4 79 5647 innerTail19_43 best.bound = innerCandidates19_42 := by
  unfold candidates
  have row : edge 5647 = SearchData.row5647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_42 : scan edge (search edge 3) 4 [79] 79 innerTail19_42 best = best := by
  rw [innerTail19_42, scan, if_neg (by decide), innerCandidates19_42_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_43
    (if 3 ≤ innerCandidates19_42.length then search edge 3 [79, 5647] innerCandidates19_42 5726 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_43

def innerTail19_41 : List Nat := 5581 :: innerTail19_42
def innerCandidates19_41 : List Nat := [5653, 7561]
set_option maxRecDepth 4096 in
theorem innerCandidates19_41_checked : candidates edge 4 79 5581 innerTail19_42 best.bound = innerCandidates19_41 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_41 : scan edge (search edge 3) 4 [79] 79 innerTail19_41 best = best := by
  rw [innerTail19_41, scan, if_neg (by decide), innerCandidates19_41_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_42
    (if 3 ≤ innerCandidates19_41.length then search edge 3 [79, 5581] innerCandidates19_41 5660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_42

def innerTail19_40 : List Nat := 5503 :: innerTail19_41
def innerCandidates19_40 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_40_checked : candidates edge 4 79 5503 innerTail19_41 best.bound = innerCandidates19_40 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_40 : scan edge (search edge 3) 4 [79] 79 innerTail19_40 best = best := by
  rw [innerTail19_40, scan, if_neg (by decide), innerCandidates19_40_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_41
    (if 3 ≤ innerCandidates19_40.length then search edge 3 [79, 5503] innerCandidates19_40 5582 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_41

def innerTail19_39 : List Nat := 5449 :: innerTail19_40
def innerCandidates19_39 : List Nat := [7549]
set_option maxRecDepth 4096 in
theorem innerCandidates19_39_checked : candidates edge 4 79 5449 innerTail19_40 best.bound = innerCandidates19_39 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_39 : scan edge (search edge 3) 4 [79] 79 innerTail19_39 best = best := by
  rw [innerTail19_39, scan, if_neg (by decide), innerCandidates19_39_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_40
    (if 3 ≤ innerCandidates19_39.length then search edge 3 [79, 5449] innerCandidates19_39 5528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_40

def innerTail19_38 : List Nat := 5323 :: innerTail19_39
def innerCandidates19_38 : List Nat := [5659]
set_option maxRecDepth 4096 in
theorem innerCandidates19_38_checked : candidates edge 4 79 5323 innerTail19_39 best.bound = innerCandidates19_38 := by
  unfold candidates
  have row : edge 5323 = SearchData.row5323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_38 : scan edge (search edge 3) 4 [79] 79 innerTail19_38 best = best := by
  rw [innerTail19_38, scan, if_neg (by decide), innerCandidates19_38_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_39
    (if 3 ≤ innerCandidates19_38.length then search edge 3 [79, 5323] innerCandidates19_38 5402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_39

def innerTail19_37 : List Nat := 5101 :: innerTail19_38
def innerCandidates19_37 : List Nat := [8263]
set_option maxRecDepth 4096 in
theorem innerCandidates19_37_checked : candidates edge 4 79 5101 innerTail19_38 best.bound = innerCandidates19_37 := by
  unfold candidates
  have row : edge 5101 = SearchData.row5101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_37 : scan edge (search edge 3) 4 [79] 79 innerTail19_37 best = best := by
  rw [innerTail19_37, scan, if_neg (by decide), innerCandidates19_37_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_38
    (if 3 ≤ innerCandidates19_37.length then search edge 3 [79, 5101] innerCandidates19_37 5180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_38

def innerTail19_36 : List Nat := 5077 :: innerTail19_37
def innerCandidates19_36 : List Nat := [6091]
set_option maxRecDepth 4096 in
theorem innerCandidates19_36_checked : candidates edge 4 79 5077 innerTail19_37 best.bound = innerCandidates19_36 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_36 : scan edge (search edge 3) 4 [79] 79 innerTail19_36 best = best := by
  rw [innerTail19_36, scan, if_neg (by decide), innerCandidates19_36_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_37
    (if 3 ≤ innerCandidates19_36.length then search edge 3 [79, 5077] innerCandidates19_36 5156 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_37

def innerTail19_35 : List Nat := 4999 :: innerTail19_36
def innerCandidates19_35 : List Nat := [5791]
set_option maxRecDepth 4096 in
theorem innerCandidates19_35_checked : candidates edge 4 79 4999 innerTail19_36 best.bound = innerCandidates19_35 := by
  unfold candidates
  have row : edge 4999 = SearchData.row4999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_35 : scan edge (search edge 3) 4 [79] 79 innerTail19_35 best = best := by
  rw [innerTail19_35, scan, if_neg (by decide), innerCandidates19_35_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_36
    (if 3 ≤ innerCandidates19_35.length then search edge 3 [79, 4999] innerCandidates19_35 5078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_36

def innerTail19_34 : List Nat := 4831 :: innerTail19_35
def innerCandidates19_34 : List Nat := [6451, 6553]
set_option maxRecDepth 4096 in
theorem innerCandidates19_34_checked : candidates edge 4 79 4831 innerTail19_35 best.bound = innerCandidates19_34 := by
  unfold candidates
  have row : edge 4831 = SearchData.row4831.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_34 : scan edge (search edge 3) 4 [79] 79 innerTail19_34 best = best := by
  rw [innerTail19_34, scan, if_neg (by decide), innerCandidates19_34_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_35
    (if 3 ≤ innerCandidates19_34.length then search edge 3 [79, 4831] innerCandidates19_34 4910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_35

def innerTail19_33 : List Nat := 4483 :: innerTail19_34
def innerCandidates19_33 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_33_checked : candidates edge 4 79 4483 innerTail19_34 best.bound = innerCandidates19_33 := by
  unfold candidates
  have row : edge 4483 = SearchData.row4483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_33 : scan edge (search edge 3) 4 [79] 79 innerTail19_33 best = best := by
  rw [innerTail19_33, scan, if_neg (by decide), innerCandidates19_33_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_34
    (if 3 ≤ innerCandidates19_33.length then search edge 3 [79, 4483] innerCandidates19_33 4562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_34

def innerTail19_32 : List Nat := 4231 :: innerTail19_33
def innerCandidates19_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_32_checked : candidates edge 4 79 4231 innerTail19_33 best.bound = innerCandidates19_32 := by
  unfold candidates
  have row : edge 4231 = SearchData.row4231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_32 : scan edge (search edge 3) 4 [79] 79 innerTail19_32 best = best := by
  rw [innerTail19_32, scan, if_neg (by decide), innerCandidates19_32_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_33
    (if 3 ≤ innerCandidates19_32.length then search edge 3 [79, 4231] innerCandidates19_32 4310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_33

def innerTail19_31 : List Nat := 4153 :: innerTail19_32
def innerCandidates19_31 : List Nat := [5077, 5101, 5503]
set_option maxRecDepth 4096 in
theorem innerCandidates19_31_checked : candidates edge 4 79 4153 innerTail19_32 best.bound = innerCandidates19_31 := by
  unfold candidates
  have row : edge 4153 = SearchData.row4153.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild19_31 : search edge 3 [79, 4153] innerCandidates19_31 4232 best = best := by decide

theorem innerScan19_31 : scan edge (search edge 3) 4 [79] 79 innerTail19_31 best = best := by
  rw [innerTail19_31, scan, if_neg (by decide), innerCandidates19_31_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_32
    (if 3 ≤ innerCandidates19_31.length then search edge 3 [79, 4153] innerCandidates19_31 4232 best else best) = best
  rw [if_pos (by decide), innerChild19_31]
  exact innerScan19_32

def innerTail19_30 : List Nat := 3853 :: innerTail19_31
def innerCandidates19_30 : List Nat := [7591]
set_option maxRecDepth 4096 in
theorem innerCandidates19_30_checked : candidates edge 4 79 3853 innerTail19_31 best.bound = innerCandidates19_30 := by
  unfold candidates
  have row : edge 3853 = SearchData.row3853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_30 : scan edge (search edge 3) 4 [79] 79 innerTail19_30 best = best := by
  rw [innerTail19_30, scan, if_neg (by decide), innerCandidates19_30_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_31
    (if 3 ≤ innerCandidates19_30.length then search edge 3 [79, 3853] innerCandidates19_30 3932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_31

def innerTail19_29 : List Nat := 3733 :: innerTail19_30
def innerCandidates19_29 : List Nat := [6451]
set_option maxRecDepth 4096 in
theorem innerCandidates19_29_checked : candidates edge 4 79 3733 innerTail19_30 best.bound = innerCandidates19_29 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_29 : scan edge (search edge 3) 4 [79] 79 innerTail19_29 best = best := by
  rw [innerTail19_29, scan, if_neg (by decide), innerCandidates19_29_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_30
    (if 3 ≤ innerCandidates19_29.length then search edge 3 [79, 3733] innerCandidates19_29 3812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_30

def innerTail19_28 : List Nat := 3607 :: innerTail19_29
def innerCandidates19_28 : List Nat := [5077]
set_option maxRecDepth 4096 in
theorem innerCandidates19_28_checked : candidates edge 4 79 3607 innerTail19_29 best.bound = innerCandidates19_28 := by
  unfold candidates
  have row : edge 3607 = SearchData.row3607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_28 : scan edge (search edge 3) 4 [79] 79 innerTail19_28 best = best := by
  rw [innerTail19_28, scan, if_neg (by decide), innerCandidates19_28_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_29
    (if 3 ≤ innerCandidates19_28.length then search edge 3 [79, 3607] innerCandidates19_28 3686 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_29

def innerTail19_27 : List Nat := 3547 :: innerTail19_28
def innerCandidates19_27 : List Nat := [5449]
set_option maxRecDepth 4096 in
theorem innerCandidates19_27_checked : candidates edge 4 79 3547 innerTail19_28 best.bound = innerCandidates19_27 := by
  unfold candidates
  have row : edge 3547 = SearchData.row3547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_27 : scan edge (search edge 3) 4 [79] 79 innerTail19_27 best = best := by
  rw [innerTail19_27, scan, if_neg (by decide), innerCandidates19_27_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_28
    (if 3 ≤ innerCandidates19_27.length then search edge 3 [79, 3547] innerCandidates19_27 3626 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_28

def innerTail19_26 : List Nat := 3517 :: innerTail19_27
def innerCandidates19_26 : List Nat := [5659, 8263]
set_option maxRecDepth 4096 in
theorem innerCandidates19_26_checked : candidates edge 4 79 3517 innerTail19_27 best.bound = innerCandidates19_26 := by
  unfold candidates
  have row : edge 3517 = SearchData.row3517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_26 : scan edge (search edge 3) 4 [79] 79 innerTail19_26 best = best := by
  rw [innerTail19_26, scan, if_neg (by decide), innerCandidates19_26_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_27
    (if 3 ≤ innerCandidates19_26.length then search edge 3 [79, 3517] innerCandidates19_26 3596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_27

def innerTail19_25 : List Nat := 3511 :: innerTail19_26
def innerCandidates19_25 : List Nat := [4153, 7561]
set_option maxRecDepth 4096 in
theorem innerCandidates19_25_checked : candidates edge 4 79 3511 innerTail19_26 best.bound = innerCandidates19_25 := by
  unfold candidates
  have row : edge 3511 = SearchData.row3511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_25 : scan edge (search edge 3) 4 [79] 79 innerTail19_25 best = best := by
  rw [innerTail19_25, scan, if_neg (by decide), innerCandidates19_25_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_26
    (if 3 ≤ innerCandidates19_25.length then search edge 3 [79, 3511] innerCandidates19_25 3590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_26

def innerTail19_24 : List Nat := 3343 :: innerTail19_25
def innerCandidates19_24 : List Nat := [4483, 5449, 5791, 6373, 6451]
set_option maxRecDepth 4096 in
theorem innerCandidates19_24_checked : candidates edge 4 79 3343 innerTail19_25 best.bound = innerCandidates19_24 := by
  unfold candidates
  have row : edge 3343 = SearchData.row3343.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild19_24 : search edge 3 [79, 3343] innerCandidates19_24 3422 best = best := by decide

theorem innerScan19_24 : scan edge (search edge 3) 4 [79] 79 innerTail19_24 best = best := by
  rw [innerTail19_24, scan, if_neg (by decide), innerCandidates19_24_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_25
    (if 3 ≤ innerCandidates19_24.length then search edge 3 [79, 3343] innerCandidates19_24 3422 best else best) = best
  rw [if_pos (by decide), innerChild19_24]
  exact innerScan19_25

def innerTail19_23 : List Nat := 3253 :: innerTail19_24
def innerCandidates19_23 : List Nat := [3517, 3547, 3607, 4231, 8263]
set_option maxRecDepth 4096 in
theorem innerCandidates19_23_checked : candidates edge 4 79 3253 innerTail19_24 best.bound = innerCandidates19_23 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild19_23 : search edge 3 [79, 3253] innerCandidates19_23 3332 best = best := by decide

theorem innerScan19_23 : scan edge (search edge 3) 4 [79] 79 innerTail19_23 best = best := by
  rw [innerTail19_23, scan, if_neg (by decide), innerCandidates19_23_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_24
    (if 3 ≤ innerCandidates19_23.length then search edge 3 [79, 3253] innerCandidates19_23 3332 best else best) = best
  rw [if_pos (by decide), innerChild19_23]
  exact innerScan19_24

def innerTail19_22 : List Nat := 3181 :: innerTail19_23
def innerCandidates19_22 : List Nat := [3343]
set_option maxRecDepth 4096 in
theorem innerCandidates19_22_checked : candidates edge 4 79 3181 innerTail19_23 best.bound = innerCandidates19_22 := by
  unfold candidates
  have row : edge 3181 = SearchData.row3181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_22 : scan edge (search edge 3) 4 [79] 79 innerTail19_22 best = best := by
  rw [innerTail19_22, scan, if_neg (by decide), innerCandidates19_22_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_23
    (if 3 ≤ innerCandidates19_22.length then search edge 3 [79, 3181] innerCandidates19_22 3260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_23

def innerTail19_21 : List Nat := 2797 :: innerTail19_22
def innerCandidates19_21 : List Nat := [5449, 7207]
set_option maxRecDepth 4096 in
theorem innerCandidates19_21_checked : candidates edge 4 79 2797 innerTail19_22 best.bound = innerCandidates19_21 := by
  unfold candidates
  have row : edge 2797 = SearchData.row2797.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_21 : scan edge (search edge 3) 4 [79] 79 innerTail19_21 best = best := by
  rw [innerTail19_21, scan, if_neg (by decide), innerCandidates19_21_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_22
    (if 3 ≤ innerCandidates19_21.length then search edge 3 [79, 2797] innerCandidates19_21 2876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_22

def innerTail19_20 : List Nat := 2689 :: innerTail19_21
def innerCandidates19_20 : List Nat := [3343]
set_option maxRecDepth 4096 in
theorem innerCandidates19_20_checked : candidates edge 4 79 2689 innerTail19_21 best.bound = innerCandidates19_20 := by
  unfold candidates
  have row : edge 2689 = SearchData.row2689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_20 : scan edge (search edge 3) 4 [79] 79 innerTail19_20 best = best := by
  rw [innerTail19_20, scan, if_neg (by decide), innerCandidates19_20_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_21
    (if 3 ≤ innerCandidates19_20.length then search edge 3 [79, 2689] innerCandidates19_20 2768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_21

def innerTail19_19 : List Nat := 2647 :: innerTail19_20
def innerCandidates19_19 : List Nat := [6091, 6553, 6673]
set_option maxRecDepth 4096 in
theorem innerCandidates19_19_checked : candidates edge 4 79 2647 innerTail19_20 best.bound = innerCandidates19_19 := by
  unfold candidates
  have row : edge 2647 = PrimePairCache.smallRow2647.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild19_19 : search edge 3 [79, 2647] innerCandidates19_19 2726 best = best := by decide

theorem innerScan19_19 : scan edge (search edge 3) 4 [79] 79 innerTail19_19 best = best := by
  rw [innerTail19_19, scan, if_neg (by decide), innerCandidates19_19_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_20
    (if 3 ≤ innerCandidates19_19.length then search edge 3 [79, 2647] innerCandidates19_19 2726 best else best) = best
  rw [if_pos (by decide), innerChild19_19]
  exact innerScan19_20

def innerTail19_18 : List Nat := 2593 :: innerTail19_19
def innerCandidates19_18 : List Nat := [5101]
set_option maxRecDepth 4096 in
theorem innerCandidates19_18_checked : candidates edge 4 79 2593 innerTail19_19 best.bound = innerCandidates19_18 := by
  unfold candidates
  have row : edge 2593 = PrimePairCache.smallRow2593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_18 : scan edge (search edge 3) 4 [79] 79 innerTail19_18 best = best := by
  rw [innerTail19_18, scan, if_neg (by decide), innerCandidates19_18_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_19
    (if 3 ≤ innerCandidates19_18.length then search edge 3 [79, 2593] innerCandidates19_18 2672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_19

def innerTail19_17 : List Nat := 2551 :: innerTail19_18
def innerCandidates19_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_17_checked : candidates edge 4 79 2551 innerTail19_18 best.bound = innerCandidates19_17 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_17 : scan edge (search edge 3) 4 [79] 79 innerTail19_17 best = best := by
  rw [innerTail19_17, scan, if_neg (by decide), innerCandidates19_17_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_18
    (if 3 ≤ innerCandidates19_17.length then search edge 3 [79, 2551] innerCandidates19_17 2630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_18

def innerTail19_16 : List Nat := 2371 :: innerTail19_17
def innerCandidates19_16 : List Nat := [3607]
set_option maxRecDepth 4096 in
theorem innerCandidates19_16_checked : candidates edge 4 79 2371 innerTail19_17 best.bound = innerCandidates19_16 := by
  unfold candidates
  have row : edge 2371 = PrimePairCache.smallRow2371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_16 : scan edge (search edge 3) 4 [79] 79 innerTail19_16 best = best := by
  rw [innerTail19_16, scan, if_neg (by decide), innerCandidates19_16_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_17
    (if 3 ≤ innerCandidates19_16.length then search edge 3 [79, 2371] innerCandidates19_16 2450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_17

def innerTail19_15 : List Nat := 2179 :: innerTail19_16
def innerCandidates19_15 : List Nat := [3253, 5647]
set_option maxRecDepth 4096 in
theorem innerCandidates19_15_checked : candidates edge 4 79 2179 innerTail19_16 best.bound = innerCandidates19_15 := by
  unfold candidates
  have row : edge 2179 = PrimePairCache.smallRow2179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_15 : scan edge (search edge 3) 4 [79] 79 innerTail19_15 best = best := by
  rw [innerTail19_15, scan, if_neg (by decide), innerCandidates19_15_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_16
    (if 3 ≤ innerCandidates19_15.length then search edge 3 [79, 2179] innerCandidates19_15 2258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_16

def innerTail19_14 : List Nat := 1993 :: innerTail19_15
def innerCandidates19_14 : List Nat := [4483, 5581, 6781]
set_option maxRecDepth 4096 in
theorem innerCandidates19_14_checked : candidates edge 4 79 1993 innerTail19_15 best.bound = innerCandidates19_14 := by
  unfold candidates
  have row : edge 1993 = PrimePairCache.smallRow1993.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild19_14 : search edge 3 [79, 1993] innerCandidates19_14 2072 best = best := by decide

theorem innerScan19_14 : scan edge (search edge 3) 4 [79] 79 innerTail19_14 best = best := by
  rw [innerTail19_14, scan, if_neg (by decide), innerCandidates19_14_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_15
    (if 3 ≤ innerCandidates19_14.length then search edge 3 [79, 1993] innerCandidates19_14 2072 best else best) = best
  rw [if_pos (by decide), innerChild19_14]
  exact innerScan19_15

def innerTail19_13 : List Nat := 1933 :: innerTail19_14
def innerCandidates19_13 : List Nat := [3733, 5791, 7933]
set_option maxRecDepth 4096 in
theorem innerCandidates19_13_checked : candidates edge 4 79 1933 innerTail19_14 best.bound = innerCandidates19_13 := by
  unfold candidates
  have row : edge 1933 = PrimePairCache.smallRow1933.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild19_13 : search edge 3 [79, 1933] innerCandidates19_13 2012 best = best := by decide

theorem innerScan19_13 : scan edge (search edge 3) 4 [79] 79 innerTail19_13 best = best := by
  rw [innerTail19_13, scan, if_neg (by decide), innerCandidates19_13_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_14
    (if 3 ≤ innerCandidates19_13.length then search edge 3 [79, 1933] innerCandidates19_13 2012 best else best) = best
  rw [if_pos (by decide), innerChild19_13]
  exact innerScan19_14

def innerTail19_12 : List Nat := 1801 :: innerTail19_13
def innerCandidates19_12 : List Nat := [2647, 3253, 3547]
set_option maxRecDepth 4096 in
theorem innerCandidates19_12_checked : candidates edge 4 79 1801 innerTail19_13 best.bound = innerCandidates19_12 := by
  unfold candidates
  have row : edge 1801 = PrimePairCache.smallRow1801.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild19_12 : search edge 3 [79, 1801] innerCandidates19_12 1880 best = best := by decide

theorem innerScan19_12 : scan edge (search edge 3) 4 [79] 79 innerTail19_12 best = best := by
  rw [innerTail19_12, scan, if_neg (by decide), innerCandidates19_12_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_13
    (if 3 ≤ innerCandidates19_12.length then search edge 3 [79, 1801] innerCandidates19_12 1880 best else best) = best
  rw [if_pos (by decide), innerChild19_12]
  exact innerScan19_13

def innerTail19_11 : List Nat := 1657 :: innerTail19_12
def innerCandidates19_11 : List Nat := [5101]
set_option maxRecDepth 4096 in
theorem innerCandidates19_11_checked : candidates edge 4 79 1657 innerTail19_12 best.bound = innerCandidates19_11 := by
  unfold candidates
  have row : edge 1657 = PrimePairCache.smallRow1657.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_11 : scan edge (search edge 3) 4 [79] 79 innerTail19_11 best = best := by
  rw [innerTail19_11, scan, if_neg (by decide), innerCandidates19_11_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_12
    (if 3 ≤ innerCandidates19_11.length then search edge 3 [79, 1657] innerCandidates19_11 1736 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_12

def innerTail19_10 : List Nat := 1627 :: innerTail19_11
def innerCandidates19_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates19_10_checked : candidates edge 4 79 1627 innerTail19_11 best.bound = innerCandidates19_10 := by
  unfold candidates
  have row : edge 1627 = PrimePairCache.smallRow1627.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_10 : scan edge (search edge 3) 4 [79] 79 innerTail19_10 best = best := by
  rw [innerTail19_10, scan, if_neg (by decide), innerCandidates19_10_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_11
    (if 3 ≤ innerCandidates19_10.length then search edge 3 [79, 1627] innerCandidates19_10 1706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_11

def innerTail19_9 : List Nat := 1447 :: innerTail19_10
def innerCandidates19_9 : List Nat := [1993]
set_option maxRecDepth 4096 in
theorem innerCandidates19_9_checked : candidates edge 4 79 1447 innerTail19_10 best.bound = innerCandidates19_9 := by
  unfold candidates
  have row : edge 1447 = PrimePairCache.smallRow1447.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_9 : scan edge (search edge 3) 4 [79] 79 innerTail19_9 best = best := by
  rw [innerTail19_9, scan, if_neg (by decide), innerCandidates19_9_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_10
    (if 3 ≤ innerCandidates19_9.length then search edge 3 [79, 1447] innerCandidates19_9 1526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_10

def innerTail19_8 : List Nat := 1117 :: innerTail19_9
def innerCandidates19_8 : List Nat := [2797, 3343, 3511, 5077, 5323, 7591]
set_option maxRecDepth 4096 in
theorem innerCandidates19_8_checked : candidates edge 4 79 1117 innerTail19_9 best.bound = innerCandidates19_8 := by
  unfold candidates
  have row : edge 1117 = PrimePairCache.smallRow1117.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild19_8 : search edge 3 [79, 1117] innerCandidates19_8 1196 best = best := by decide

theorem innerScan19_8 : scan edge (search edge 3) 4 [79] 79 innerTail19_8 best = best := by
  rw [innerTail19_8, scan, if_neg (by decide), innerCandidates19_8_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_9
    (if 3 ≤ innerCandidates19_8.length then search edge 3 [79, 1117] innerCandidates19_8 1196 best else best) = best
  rw [if_pos (by decide), innerChild19_8]
  exact innerScan19_9

def innerTail19_7 : List Nat := 1093 :: innerTail19_8
def innerCandidates19_7 : List Nat := [6553, 7057]
set_option maxRecDepth 4096 in
theorem innerCandidates19_7_checked : candidates edge 4 79 1093 innerTail19_8 best.bound = innerCandidates19_7 := by
  unfold candidates
  have row : edge 1093 = PrimePairCache.smallRow1093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_7 : scan edge (search edge 3) 4 [79] 79 innerTail19_7 best = best := by
  rw [innerTail19_7, scan, if_neg (by decide), innerCandidates19_7_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_8
    (if 3 ≤ innerCandidates19_7.length then search edge 3 [79, 1093] innerCandidates19_7 1172 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_8

def innerTail19_6 : List Nat := 967 :: innerTail19_7
def innerCandidates19_6 : List Nat := [1117, 3511]
set_option maxRecDepth 4096 in
theorem innerCandidates19_6_checked : candidates edge 4 79 967 innerTail19_7 best.bound = innerCandidates19_6 := by
  unfold candidates
  have row : edge 967 = PrimePairCache.smallRow967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_6 : scan edge (search edge 3) 4 [79] 79 innerTail19_6 best = best := by
  rw [innerTail19_6, scan, if_neg (by decide), innerCandidates19_6_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_7
    (if 3 ≤ innerCandidates19_6.length then search edge 3 [79, 967] innerCandidates19_6 1046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_7

def innerTail19_5 : List Nat := 631 :: innerTail19_6
def innerCandidates19_5 : List Nat := [3253]
set_option maxRecDepth 4096 in
theorem innerCandidates19_5_checked : candidates edge 4 79 631 innerTail19_6 best.bound = innerCandidates19_5 := by
  unfold candidates
  have row : edge 631 = PrimePairCache.smallRow631.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_5 : scan edge (search edge 3) 4 [79] 79 innerTail19_5 best = best := by
  rw [innerTail19_5, scan, if_neg (by decide), innerCandidates19_5_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_6
    (if 3 ≤ innerCandidates19_5.length then search edge 3 [79, 631] innerCandidates19_5 710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_6

def innerTail19_4 : List Nat := 613 :: innerTail19_5
def innerCandidates19_4 : List Nat := [4153, 5581, 6673]
set_option maxRecDepth 4096 in
theorem innerCandidates19_4_checked : candidates edge 4 79 613 innerTail19_5 best.bound = innerCandidates19_4 := by
  unfold candidates
  have row : edge 613 = PrimePairCache.smallRow613.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild19_4 : search edge 3 [79, 613] innerCandidates19_4 692 best = best := by decide

theorem innerScan19_4 : scan edge (search edge 3) 4 [79] 79 innerTail19_4 best = best := by
  rw [innerTail19_4, scan, if_neg (by decide), innerCandidates19_4_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_5
    (if 3 ≤ innerCandidates19_4.length then search edge 3 [79, 613] innerCandidates19_4 692 best else best) = best
  rw [if_pos (by decide), innerChild19_4]
  exact innerScan19_5

def innerTail19_3 : List Nat := 397 :: innerTail19_4
def innerCandidates19_3 : List Nat := [2593, 3511, 4483, 5791, 7933]
set_option maxRecDepth 4096 in
theorem innerCandidates19_3_checked : candidates edge 4 79 397 innerTail19_4 best.bound = innerCandidates19_3 := by
  unfold candidates
  have row : edge 397 = PrimePairCache.smallRow397.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild19_3 : search edge 3 [79, 397] innerCandidates19_3 476 best = best := by decide

theorem innerScan19_3 : scan edge (search edge 3) 4 [79] 79 innerTail19_3 best = best := by
  rw [innerTail19_3, scan, if_neg (by decide), innerCandidates19_3_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_4
    (if 3 ≤ innerCandidates19_3.length then search edge 3 [79, 397] innerCandidates19_3 476 best else best) = best
  rw [if_pos (by decide), innerChild19_3]
  exact innerScan19_4

def innerTail19_2 : List Nat := 367 :: innerTail19_3
def innerCandidates19_2 : List Nat := [613, 1117, 6091, 7207]
set_option maxRecDepth 4096 in
theorem innerCandidates19_2_checked : candidates edge 4 79 367 innerTail19_3 best.bound = innerCandidates19_2 := by
  unfold candidates
  have row : edge 367 = PrimePairCache.smallRow367.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild19_2 : search edge 3 [79, 367] innerCandidates19_2 446 best = best := by decide

theorem innerScan19_2 : scan edge (search edge 3) 4 [79] 79 innerTail19_2 best = best := by
  rw [innerTail19_2, scan, if_neg (by decide), innerCandidates19_2_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_3
    (if 3 ≤ innerCandidates19_2.length then search edge 3 [79, 367] innerCandidates19_2 446 best else best) = best
  rw [if_pos (by decide), innerChild19_2]
  exact innerScan19_3

def innerTail19_1 : List Nat := 241 :: innerTail19_2
def innerCandidates19_1 : List Nat := [2797, 3253, 3511, 5503]
set_option maxRecDepth 4096 in
theorem innerCandidates19_1_checked : candidates edge 4 79 241 innerTail19_2 best.bound = innerCandidates19_1 := by
  unfold candidates
  have row : edge 241 = PrimePairCache.smallRow241.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild19_1 : search edge 3 [79, 241] innerCandidates19_1 320 best = best := by decide

theorem innerScan19_1 : scan edge (search edge 3) 4 [79] 79 innerTail19_1 best = best := by
  rw [innerTail19_1, scan, if_neg (by decide), innerCandidates19_1_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_2
    (if 3 ≤ innerCandidates19_1.length then search edge 3 [79, 241] innerCandidates19_1 320 best else best) = best
  rw [if_pos (by decide), innerChild19_1]
  exact innerScan19_2

def innerTail19_0 : List Nat := 193 :: innerTail19_1
def innerCandidates19_0 : List Nat := [1627]
set_option maxRecDepth 4096 in
theorem innerCandidates19_0_checked : candidates edge 4 79 193 innerTail19_1 best.bound = innerCandidates19_0 := by
  unfold candidates
  have row : edge 193 = PrimePairCache.smallRow193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan19_0 : scan edge (search edge 3) 4 [79] 79 innerTail19_0 best = best := by
  rw [innerTail19_0, scan, if_neg (by decide), innerCandidates19_0_checked]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_1
    (if 3 ≤ innerCandidates19_0.length then search edge 3 [79, 193] innerCandidates19_0 272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan19_1

theorem rootBranch19 : search edge 4 [79] rootCandidates19 79 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [79] 79 innerTail19_0 best = best
  exact innerScan19_0

end ProjectEulerSolutions.PrimePairUpper.Search
