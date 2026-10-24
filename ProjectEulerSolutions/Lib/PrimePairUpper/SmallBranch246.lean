import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates21 : List Nat := [107, 137, 293, 431, 443, 521, 809, 821, 977, 983, 1061, 1223, 1409, 1427, 1523, 1571, 1823, 2141, 2153, 2267, 2357, 2579, 2777, 3407, 3671, 3821, 3917, 3989, 4211, 4259, 4409, 4721, 4973, 5003, 5039, 5087, 5171, 5189, 5309, 5333, 5387, 5393, 5507, 5651, 5861, 5879, 5903, 5987, 6113, 6263, 6299, 6323, 6353, 6521, 6947, 7103, 7307, 7649, 7727, 8147]
set_option maxRecDepth 4096 in
theorem rootCandidates21_checked : candidates edge 5 0 89 tail22 best.bound = rootCandidates21 := by
  unfold candidates
  have row : edge 89 = PrimePairCache.smallRow89.contains := by funext q; rfl
  rw [row]
  decide

def innerTail21_60 : List Nat := []
theorem innerScan21_60 : scan edge (search edge 3) 4 [89] 89 innerTail21_60 best = best := rfl

def innerTail21_59 : List Nat := 8147 :: innerTail21_60
def innerCandidates21_59 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_59_checked : candidates edge 4 89 8147 innerTail21_60 best.bound = innerCandidates21_59 := by
  unfold candidates
  have row : edge 8147 = SearchData.row8147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_59 : scan edge (search edge 3) 4 [89] 89 innerTail21_59 best = best := by
  rw [innerTail21_59, scan, if_neg (by decide), innerCandidates21_59_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_60
    (if 3 ≤ innerCandidates21_59.length then search edge 3 [89, 8147] innerCandidates21_59 8236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_60

def innerTail21_58 : List Nat := 7727 :: innerTail21_59
def innerCandidates21_58 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_58_checked : candidates edge 4 89 7727 innerTail21_59 best.bound = innerCandidates21_58 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_58 : scan edge (search edge 3) 4 [89] 89 innerTail21_58 best = best := by
  rw [innerTail21_58, scan, if_neg (by decide), innerCandidates21_58_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_59
    (if 3 ≤ innerCandidates21_58.length then search edge 3 [89, 7727] innerCandidates21_58 7816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_59

def innerTail21_57 : List Nat := 7649 :: innerTail21_58
def innerCandidates21_57 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_57_checked : candidates edge 4 89 7649 innerTail21_58 best.bound = innerCandidates21_57 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_57 : scan edge (search edge 3) 4 [89] 89 innerTail21_57 best = best := by
  rw [innerTail21_57, scan, if_neg (by decide), innerCandidates21_57_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_58
    (if 3 ≤ innerCandidates21_57.length then search edge 3 [89, 7649] innerCandidates21_57 7738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_58

def innerTail21_56 : List Nat := 7307 :: innerTail21_57
def innerCandidates21_56 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_56_checked : candidates edge 4 89 7307 innerTail21_57 best.bound = innerCandidates21_56 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_56 : scan edge (search edge 3) 4 [89] 89 innerTail21_56 best = best := by
  rw [innerTail21_56, scan, if_neg (by decide), innerCandidates21_56_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_57
    (if 3 ≤ innerCandidates21_56.length then search edge 3 [89, 7307] innerCandidates21_56 7396 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_57

def innerTail21_55 : List Nat := 7103 :: innerTail21_56
def innerCandidates21_55 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_55_checked : candidates edge 4 89 7103 innerTail21_56 best.bound = innerCandidates21_55 := by
  unfold candidates
  have row : edge 7103 = SearchData.row7103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_55 : scan edge (search edge 3) 4 [89] 89 innerTail21_55 best = best := by
  rw [innerTail21_55, scan, if_neg (by decide), innerCandidates21_55_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_56
    (if 3 ≤ innerCandidates21_55.length then search edge 3 [89, 7103] innerCandidates21_55 7192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_56

def innerTail21_54 : List Nat := 6947 :: innerTail21_55
def innerCandidates21_54 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_54_checked : candidates edge 4 89 6947 innerTail21_55 best.bound = innerCandidates21_54 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_54 : scan edge (search edge 3) 4 [89] 89 innerTail21_54 best = best := by
  rw [innerTail21_54, scan, if_neg (by decide), innerCandidates21_54_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_55
    (if 3 ≤ innerCandidates21_54.length then search edge 3 [89, 6947] innerCandidates21_54 7036 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_55

def innerTail21_53 : List Nat := 6521 :: innerTail21_54
def innerCandidates21_53 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_53_checked : candidates edge 4 89 6521 innerTail21_54 best.bound = innerCandidates21_53 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_53 : scan edge (search edge 3) 4 [89] 89 innerTail21_53 best = best := by
  rw [innerTail21_53, scan, if_neg (by decide), innerCandidates21_53_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_54
    (if 3 ≤ innerCandidates21_53.length then search edge 3 [89, 6521] innerCandidates21_53 6610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_54

def innerTail21_52 : List Nat := 6353 :: innerTail21_53
def innerCandidates21_52 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_52_checked : candidates edge 4 89 6353 innerTail21_53 best.bound = innerCandidates21_52 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_52 : scan edge (search edge 3) 4 [89] 89 innerTail21_52 best = best := by
  rw [innerTail21_52, scan, if_neg (by decide), innerCandidates21_52_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_53
    (if 3 ≤ innerCandidates21_52.length then search edge 3 [89, 6353] innerCandidates21_52 6442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_53

def innerTail21_51 : List Nat := 6323 :: innerTail21_52
def innerCandidates21_51 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_51_checked : candidates edge 4 89 6323 innerTail21_52 best.bound = innerCandidates21_51 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_51 : scan edge (search edge 3) 4 [89] 89 innerTail21_51 best = best := by
  rw [innerTail21_51, scan, if_neg (by decide), innerCandidates21_51_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_52
    (if 3 ≤ innerCandidates21_51.length then search edge 3 [89, 6323] innerCandidates21_51 6412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_52

def innerTail21_50 : List Nat := 6299 :: innerTail21_51
def innerCandidates21_50 : List Nat := [8147]
set_option maxRecDepth 4096 in
theorem innerCandidates21_50_checked : candidates edge 4 89 6299 innerTail21_51 best.bound = innerCandidates21_50 := by
  unfold candidates
  have row : edge 6299 = SearchData.row6299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_50 : scan edge (search edge 3) 4 [89] 89 innerTail21_50 best = best := by
  rw [innerTail21_50, scan, if_neg (by decide), innerCandidates21_50_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_51
    (if 3 ≤ innerCandidates21_50.length then search edge 3 [89, 6299] innerCandidates21_50 6388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_51

def innerTail21_49 : List Nat := 6263 :: innerTail21_50
def innerCandidates21_49 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_49_checked : candidates edge 4 89 6263 innerTail21_50 best.bound = innerCandidates21_49 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_49 : scan edge (search edge 3) 4 [89] 89 innerTail21_49 best = best := by
  rw [innerTail21_49, scan, if_neg (by decide), innerCandidates21_49_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_50
    (if 3 ≤ innerCandidates21_49.length then search edge 3 [89, 6263] innerCandidates21_49 6352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_50

def innerTail21_48 : List Nat := 6113 :: innerTail21_49
def innerCandidates21_48 : List Nat := [6353]
set_option maxRecDepth 4096 in
theorem innerCandidates21_48_checked : candidates edge 4 89 6113 innerTail21_49 best.bound = innerCandidates21_48 := by
  unfold candidates
  have row : edge 6113 = SearchData.row6113.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_48 : scan edge (search edge 3) 4 [89] 89 innerTail21_48 best = best := by
  rw [innerTail21_48, scan, if_neg (by decide), innerCandidates21_48_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_49
    (if 3 ≤ innerCandidates21_48.length then search edge 3 [89, 6113] innerCandidates21_48 6202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_49

def innerTail21_47 : List Nat := 5987 :: innerTail21_48
def innerCandidates21_47 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_47_checked : candidates edge 4 89 5987 innerTail21_48 best.bound = innerCandidates21_47 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_47 : scan edge (search edge 3) 4 [89] 89 innerTail21_47 best = best := by
  rw [innerTail21_47, scan, if_neg (by decide), innerCandidates21_47_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_48
    (if 3 ≤ innerCandidates21_47.length then search edge 3 [89, 5987] innerCandidates21_47 6076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_48

def innerTail21_46 : List Nat := 5903 :: innerTail21_47
def innerCandidates21_46 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_46_checked : candidates edge 4 89 5903 innerTail21_47 best.bound = innerCandidates21_46 := by
  unfold candidates
  have row : edge 5903 = SearchData.row5903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_46 : scan edge (search edge 3) 4 [89] 89 innerTail21_46 best = best := by
  rw [innerTail21_46, scan, if_neg (by decide), innerCandidates21_46_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_47
    (if 3 ≤ innerCandidates21_46.length then search edge 3 [89, 5903] innerCandidates21_46 5992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_47

def innerTail21_45 : List Nat := 5879 :: innerTail21_46
def innerCandidates21_45 : List Nat := [6263]
set_option maxRecDepth 4096 in
theorem innerCandidates21_45_checked : candidates edge 4 89 5879 innerTail21_46 best.bound = innerCandidates21_45 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_45 : scan edge (search edge 3) 4 [89] 89 innerTail21_45 best = best := by
  rw [innerTail21_45, scan, if_neg (by decide), innerCandidates21_45_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_46
    (if 3 ≤ innerCandidates21_45.length then search edge 3 [89, 5879] innerCandidates21_45 5968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_46

def innerTail21_44 : List Nat := 5861 :: innerTail21_45
def innerCandidates21_44 : List Nat := [5987]
set_option maxRecDepth 4096 in
theorem innerCandidates21_44_checked : candidates edge 4 89 5861 innerTail21_45 best.bound = innerCandidates21_44 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_44 : scan edge (search edge 3) 4 [89] 89 innerTail21_44 best = best := by
  rw [innerTail21_44, scan, if_neg (by decide), innerCandidates21_44_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_45
    (if 3 ≤ innerCandidates21_44.length then search edge 3 [89, 5861] innerCandidates21_44 5950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_45

def innerTail21_43 : List Nat := 5651 :: innerTail21_44
def innerCandidates21_43 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_43_checked : candidates edge 4 89 5651 innerTail21_44 best.bound = innerCandidates21_43 := by
  unfold candidates
  have row : edge 5651 = SearchData.row5651.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_43 : scan edge (search edge 3) 4 [89] 89 innerTail21_43 best = best := by
  rw [innerTail21_43, scan, if_neg (by decide), innerCandidates21_43_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_44
    (if 3 ≤ innerCandidates21_43.length then search edge 3 [89, 5651] innerCandidates21_43 5740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_44

def innerTail21_42 : List Nat := 5507 :: innerTail21_43
def innerCandidates21_42 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_42_checked : candidates edge 4 89 5507 innerTail21_43 best.bound = innerCandidates21_42 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_42 : scan edge (search edge 3) 4 [89] 89 innerTail21_42 best = best := by
  rw [innerTail21_42, scan, if_neg (by decide), innerCandidates21_42_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_43
    (if 3 ≤ innerCandidates21_42.length then search edge 3 [89, 5507] innerCandidates21_42 5596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_43

def innerTail21_41 : List Nat := 5393 :: innerTail21_42
def innerCandidates21_41 : List Nat := [6521]
set_option maxRecDepth 4096 in
theorem innerCandidates21_41_checked : candidates edge 4 89 5393 innerTail21_42 best.bound = innerCandidates21_41 := by
  unfold candidates
  have row : edge 5393 = SearchData.row5393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_41 : scan edge (search edge 3) 4 [89] 89 innerTail21_41 best = best := by
  rw [innerTail21_41, scan, if_neg (by decide), innerCandidates21_41_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_42
    (if 3 ≤ innerCandidates21_41.length then search edge 3 [89, 5393] innerCandidates21_41 5482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_42

def innerTail21_40 : List Nat := 5387 :: innerTail21_41
def innerCandidates21_40 : List Nat := [6353]
set_option maxRecDepth 4096 in
theorem innerCandidates21_40_checked : candidates edge 4 89 5387 innerTail21_41 best.bound = innerCandidates21_40 := by
  unfold candidates
  have row : edge 5387 = SearchData.row5387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_40 : scan edge (search edge 3) 4 [89] 89 innerTail21_40 best = best := by
  rw [innerTail21_40, scan, if_neg (by decide), innerCandidates21_40_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_41
    (if 3 ≤ innerCandidates21_40.length then search edge 3 [89, 5387] innerCandidates21_40 5476 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_41

def innerTail21_39 : List Nat := 5333 :: innerTail21_40
def innerCandidates21_39 : List Nat := [7649]
set_option maxRecDepth 4096 in
theorem innerCandidates21_39_checked : candidates edge 4 89 5333 innerTail21_40 best.bound = innerCandidates21_39 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_39 : scan edge (search edge 3) 4 [89] 89 innerTail21_39 best = best := by
  rw [innerTail21_39, scan, if_neg (by decide), innerCandidates21_39_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_40
    (if 3 ≤ innerCandidates21_39.length then search edge 3 [89, 5333] innerCandidates21_39 5422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_40

def innerTail21_38 : List Nat := 5309 :: innerTail21_39
def innerCandidates21_38 : List Nat := [8147]
set_option maxRecDepth 4096 in
theorem innerCandidates21_38_checked : candidates edge 4 89 5309 innerTail21_39 best.bound = innerCandidates21_38 := by
  unfold candidates
  have row : edge 5309 = SearchData.row5309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_38 : scan edge (search edge 3) 4 [89] 89 innerTail21_38 best = best := by
  rw [innerTail21_38, scan, if_neg (by decide), innerCandidates21_38_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_39
    (if 3 ≤ innerCandidates21_38.length then search edge 3 [89, 5309] innerCandidates21_38 5398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_39

def innerTail21_37 : List Nat := 5189 :: innerTail21_38
def innerCandidates21_37 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_37_checked : candidates edge 4 89 5189 innerTail21_38 best.bound = innerCandidates21_37 := by
  unfold candidates
  have row : edge 5189 = SearchData.row5189.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_37 : scan edge (search edge 3) 4 [89] 89 innerTail21_37 best = best := by
  rw [innerTail21_37, scan, if_neg (by decide), innerCandidates21_37_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_38
    (if 3 ≤ innerCandidates21_37.length then search edge 3 [89, 5189] innerCandidates21_37 5278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_38

def innerTail21_36 : List Nat := 5171 :: innerTail21_37
def innerCandidates21_36 : List Nat := [5393]
set_option maxRecDepth 4096 in
theorem innerCandidates21_36_checked : candidates edge 4 89 5171 innerTail21_37 best.bound = innerCandidates21_36 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_36 : scan edge (search edge 3) 4 [89] 89 innerTail21_36 best = best := by
  rw [innerTail21_36, scan, if_neg (by decide), innerCandidates21_36_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_37
    (if 3 ≤ innerCandidates21_36.length then search edge 3 [89, 5171] innerCandidates21_36 5260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_37

def innerTail21_35 : List Nat := 5087 :: innerTail21_36
def innerCandidates21_35 : List Nat := [5861, 8147]
set_option maxRecDepth 4096 in
theorem innerCandidates21_35_checked : candidates edge 4 89 5087 innerTail21_36 best.bound = innerCandidates21_35 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_35 : scan edge (search edge 3) 4 [89] 89 innerTail21_35 best = best := by
  rw [innerTail21_35, scan, if_neg (by decide), innerCandidates21_35_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_36
    (if 3 ≤ innerCandidates21_35.length then search edge 3 [89, 5087] innerCandidates21_35 5176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_36

def innerTail21_34 : List Nat := 5039 :: innerTail21_35
def innerCandidates21_34 : List Nat := [6299, 7727]
set_option maxRecDepth 4096 in
theorem innerCandidates21_34_checked : candidates edge 4 89 5039 innerTail21_35 best.bound = innerCandidates21_34 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_34 : scan edge (search edge 3) 4 [89] 89 innerTail21_34 best = best := by
  rw [innerTail21_34, scan, if_neg (by decide), innerCandidates21_34_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_35
    (if 3 ≤ innerCandidates21_34.length then search edge 3 [89, 5039] innerCandidates21_34 5128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_35

def innerTail21_33 : List Nat := 5003 :: innerTail21_34
def innerCandidates21_33 : List Nat := [5087, 8147]
set_option maxRecDepth 4096 in
theorem innerCandidates21_33_checked : candidates edge 4 89 5003 innerTail21_34 best.bound = innerCandidates21_33 := by
  unfold candidates
  have row : edge 5003 = SearchData.row5003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_33 : scan edge (search edge 3) 4 [89] 89 innerTail21_33 best = best := by
  rw [innerTail21_33, scan, if_neg (by decide), innerCandidates21_33_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_34
    (if 3 ≤ innerCandidates21_33.length then search edge 3 [89, 5003] innerCandidates21_33 5092 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_34

def innerTail21_32 : List Nat := 4973 :: innerTail21_33
def innerCandidates21_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_32_checked : candidates edge 4 89 4973 innerTail21_33 best.bound = innerCandidates21_32 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_32 : scan edge (search edge 3) 4 [89] 89 innerTail21_32 best = best := by
  rw [innerTail21_32, scan, if_neg (by decide), innerCandidates21_32_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_33
    (if 3 ≤ innerCandidates21_32.length then search edge 3 [89, 4973] innerCandidates21_32 5062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_33

def innerTail21_31 : List Nat := 4721 :: innerTail21_32
def innerCandidates21_31 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_31_checked : candidates edge 4 89 4721 innerTail21_32 best.bound = innerCandidates21_31 := by
  unfold candidates
  have row : edge 4721 = SearchData.row4721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_31 : scan edge (search edge 3) 4 [89] 89 innerTail21_31 best = best := by
  rw [innerTail21_31, scan, if_neg (by decide), innerCandidates21_31_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_32
    (if 3 ≤ innerCandidates21_31.length then search edge 3 [89, 4721] innerCandidates21_31 4810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_32

def innerTail21_30 : List Nat := 4409 :: innerTail21_31
def innerCandidates21_30 : List Nat := [5087]
set_option maxRecDepth 4096 in
theorem innerCandidates21_30_checked : candidates edge 4 89 4409 innerTail21_31 best.bound = innerCandidates21_30 := by
  unfold candidates
  have row : edge 4409 = SearchData.row4409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_30 : scan edge (search edge 3) 4 [89] 89 innerTail21_30 best = best := by
  rw [innerTail21_30, scan, if_neg (by decide), innerCandidates21_30_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_31
    (if 3 ≤ innerCandidates21_30.length then search edge 3 [89, 4409] innerCandidates21_30 4498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_31

def innerTail21_29 : List Nat := 4259 :: innerTail21_30
def innerCandidates21_29 : List Nat := [5087]
set_option maxRecDepth 4096 in
theorem innerCandidates21_29_checked : candidates edge 4 89 4259 innerTail21_30 best.bound = innerCandidates21_29 := by
  unfold candidates
  have row : edge 4259 = SearchData.row4259.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_29 : scan edge (search edge 3) 4 [89] 89 innerTail21_29 best = best := by
  rw [innerTail21_29, scan, if_neg (by decide), innerCandidates21_29_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_30
    (if 3 ≤ innerCandidates21_29.length then search edge 3 [89, 4259] innerCandidates21_29 4348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_30

def innerTail21_28 : List Nat := 4211 :: innerTail21_29
def innerCandidates21_28 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_28_checked : candidates edge 4 89 4211 innerTail21_29 best.bound = innerCandidates21_28 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_28 : scan edge (search edge 3) 4 [89] 89 innerTail21_28 best = best := by
  rw [innerTail21_28, scan, if_neg (by decide), innerCandidates21_28_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_29
    (if 3 ≤ innerCandidates21_28.length then search edge 3 [89, 4211] innerCandidates21_28 4300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_29

def innerTail21_27 : List Nat := 3989 :: innerTail21_28
def innerCandidates21_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_27_checked : candidates edge 4 89 3989 innerTail21_28 best.bound = innerCandidates21_27 := by
  unfold candidates
  have row : edge 3989 = SearchData.row3989.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_27 : scan edge (search edge 3) 4 [89] 89 innerTail21_27 best = best := by
  rw [innerTail21_27, scan, if_neg (by decide), innerCandidates21_27_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_28
    (if 3 ≤ innerCandidates21_27.length then search edge 3 [89, 3989] innerCandidates21_27 4078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_28

def innerTail21_26 : List Nat := 3917 :: innerTail21_27
def innerCandidates21_26 : List Nat := [4211, 5189, 5987, 6323]
set_option maxRecDepth 4096 in
theorem innerCandidates21_26_checked : candidates edge 4 89 3917 innerTail21_27 best.bound = innerCandidates21_26 := by
  unfold candidates
  have row : edge 3917 = SearchData.row3917.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild21_26 : search edge 3 [89, 3917] innerCandidates21_26 4006 best = best := by decide

theorem innerScan21_26 : scan edge (search edge 3) 4 [89] 89 innerTail21_26 best = best := by
  rw [innerTail21_26, scan, if_neg (by decide), innerCandidates21_26_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_27
    (if 3 ≤ innerCandidates21_26.length then search edge 3 [89, 3917] innerCandidates21_26 4006 best else best) = best
  rw [if_pos (by decide), innerChild21_26]
  exact innerScan21_27

def innerTail21_25 : List Nat := 3821 :: innerTail21_26
def innerCandidates21_25 : List Nat := [6263, 7307]
set_option maxRecDepth 4096 in
theorem innerCandidates21_25_checked : candidates edge 4 89 3821 innerTail21_26 best.bound = innerCandidates21_25 := by
  unfold candidates
  have row : edge 3821 = SearchData.row3821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_25 : scan edge (search edge 3) 4 [89] 89 innerTail21_25 best = best := by
  rw [innerTail21_25, scan, if_neg (by decide), innerCandidates21_25_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_26
    (if 3 ≤ innerCandidates21_25.length then search edge 3 [89, 3821] innerCandidates21_25 3910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_26

def innerTail21_24 : List Nat := 3671 :: innerTail21_25
def innerCandidates21_24 : List Nat := [5039]
set_option maxRecDepth 4096 in
theorem innerCandidates21_24_checked : candidates edge 4 89 3671 innerTail21_25 best.bound = innerCandidates21_24 := by
  unfold candidates
  have row : edge 3671 = SearchData.row3671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_24 : scan edge (search edge 3) 4 [89] 89 innerTail21_24 best = best := by
  rw [innerTail21_24, scan, if_neg (by decide), innerCandidates21_24_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_25
    (if 3 ≤ innerCandidates21_24.length then search edge 3 [89, 3671] innerCandidates21_24 3760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_25

def innerTail21_23 : List Nat := 3407 :: innerTail21_24
def innerCandidates21_23 : List Nat := [3671, 6299]
set_option maxRecDepth 4096 in
theorem innerCandidates21_23_checked : candidates edge 4 89 3407 innerTail21_24 best.bound = innerCandidates21_23 := by
  unfold candidates
  have row : edge 3407 = SearchData.row3407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_23 : scan edge (search edge 3) 4 [89] 89 innerTail21_23 best = best := by
  rw [innerTail21_23, scan, if_neg (by decide), innerCandidates21_23_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_24
    (if 3 ≤ innerCandidates21_23.length then search edge 3 [89, 3407] innerCandidates21_23 3496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_24

def innerTail21_22 : List Nat := 2777 :: innerTail21_23
def innerCandidates21_22 : List Nat := [4973]
set_option maxRecDepth 4096 in
theorem innerCandidates21_22_checked : candidates edge 4 89 2777 innerTail21_23 best.bound = innerCandidates21_22 := by
  unfold candidates
  have row : edge 2777 = SearchData.row2777.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_22 : scan edge (search edge 3) 4 [89] 89 innerTail21_22 best = best := by
  rw [innerTail21_22, scan, if_neg (by decide), innerCandidates21_22_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_23
    (if 3 ≤ innerCandidates21_22.length then search edge 3 [89, 2777] innerCandidates21_22 2866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_23

def innerTail21_21 : List Nat := 2579 :: innerTail21_22
def innerCandidates21_21 : List Nat := [7727]
set_option maxRecDepth 4096 in
theorem innerCandidates21_21_checked : candidates edge 4 89 2579 innerTail21_22 best.bound = innerCandidates21_21 := by
  unfold candidates
  have row : edge 2579 = PrimePairCache.smallRow2579.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_21 : scan edge (search edge 3) 4 [89] 89 innerTail21_21 best = best := by
  rw [innerTail21_21, scan, if_neg (by decide), innerCandidates21_21_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_22
    (if 3 ≤ innerCandidates21_21.length then search edge 3 [89, 2579] innerCandidates21_21 2668 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_22

def innerTail21_20 : List Nat := 2357 :: innerTail21_21
def innerCandidates21_20 : List Nat := [2579, 3821, 5087, 6323]
set_option maxRecDepth 4096 in
theorem innerCandidates21_20_checked : candidates edge 4 89 2357 innerTail21_21 best.bound = innerCandidates21_20 := by
  unfold candidates
  have row : edge 2357 = PrimePairCache.smallRow2357.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild21_20 : search edge 3 [89, 2357] innerCandidates21_20 2446 best = best := by decide

theorem innerScan21_20 : scan edge (search edge 3) 4 [89] 89 innerTail21_20 best = best := by
  rw [innerTail21_20, scan, if_neg (by decide), innerCandidates21_20_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_21
    (if 3 ≤ innerCandidates21_20.length then search edge 3 [89, 2357] innerCandidates21_20 2446 best else best) = best
  rw [if_pos (by decide), innerChild21_20]
  exact innerScan21_21

def innerTail21_19 : List Nat := 2267 :: innerTail21_20
def innerCandidates21_19 : List Nat := [4721, 5879, 8147]
set_option maxRecDepth 4096 in
theorem innerCandidates21_19_checked : candidates edge 4 89 2267 innerTail21_20 best.bound = innerCandidates21_19 := by
  unfold candidates
  have row : edge 2267 = PrimePairCache.smallRow2267.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild21_19 : search edge 3 [89, 2267] innerCandidates21_19 2356 best = best := by decide

theorem innerScan21_19 : scan edge (search edge 3) 4 [89] 89 innerTail21_19 best = best := by
  rw [innerTail21_19, scan, if_neg (by decide), innerCandidates21_19_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_20
    (if 3 ≤ innerCandidates21_19.length then search edge 3 [89, 2267] innerCandidates21_19 2356 best else best) = best
  rw [if_pos (by decide), innerChild21_19]
  exact innerScan21_20

def innerTail21_18 : List Nat := 2153 :: innerTail21_19
def innerCandidates21_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates21_18_checked : candidates edge 4 89 2153 innerTail21_19 best.bound = innerCandidates21_18 := by
  unfold candidates
  have row : edge 2153 = PrimePairCache.smallRow2153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_18 : scan edge (search edge 3) 4 [89] 89 innerTail21_18 best = best := by
  rw [innerTail21_18, scan, if_neg (by decide), innerCandidates21_18_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_19
    (if 3 ≤ innerCandidates21_18.length then search edge 3 [89, 2153] innerCandidates21_18 2242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_19

def innerTail21_17 : List Nat := 2141 :: innerTail21_18
def innerCandidates21_17 : List Nat := [2267, 7727]
set_option maxRecDepth 4096 in
theorem innerCandidates21_17_checked : candidates edge 4 89 2141 innerTail21_18 best.bound = innerCandidates21_17 := by
  unfold candidates
  have row : edge 2141 = PrimePairCache.smallRow2141.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_17 : scan edge (search edge 3) 4 [89] 89 innerTail21_17 best = best := by
  rw [innerTail21_17, scan, if_neg (by decide), innerCandidates21_17_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_18
    (if 3 ≤ innerCandidates21_17.length then search edge 3 [89, 2141] innerCandidates21_17 2230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_18

def innerTail21_16 : List Nat := 1823 :: innerTail21_17
def innerCandidates21_16 : List Nat := [6353, 6947]
set_option maxRecDepth 4096 in
theorem innerCandidates21_16_checked : candidates edge 4 89 1823 innerTail21_17 best.bound = innerCandidates21_16 := by
  unfold candidates
  have row : edge 1823 = PrimePairCache.smallRow1823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_16 : scan edge (search edge 3) 4 [89] 89 innerTail21_16 best = best := by
  rw [innerTail21_16, scan, if_neg (by decide), innerCandidates21_16_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_17
    (if 3 ≤ innerCandidates21_16.length then search edge 3 [89, 1823] innerCandidates21_16 1912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_17

def innerTail21_15 : List Nat := 1571 :: innerTail21_16
def innerCandidates21_15 : List Nat := [5333, 5879]
set_option maxRecDepth 4096 in
theorem innerCandidates21_15_checked : candidates edge 4 89 1571 innerTail21_16 best.bound = innerCandidates21_15 := by
  unfold candidates
  have row : edge 1571 = PrimePairCache.smallRow1571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_15 : scan edge (search edge 3) 4 [89] 89 innerTail21_15 best = best := by
  rw [innerTail21_15, scan, if_neg (by decide), innerCandidates21_15_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_16
    (if 3 ≤ innerCandidates21_15.length then search edge 3 [89, 1571] innerCandidates21_15 1660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_16

def innerTail21_14 : List Nat := 1523 :: innerTail21_15
def innerCandidates21_14 : List Nat := [2579, 5309]
set_option maxRecDepth 4096 in
theorem innerCandidates21_14_checked : candidates edge 4 89 1523 innerTail21_15 best.bound = innerCandidates21_14 := by
  unfold candidates
  have row : edge 1523 = PrimePairCache.smallRow1523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_14 : scan edge (search edge 3) 4 [89] 89 innerTail21_14 best = best := by
  rw [innerTail21_14, scan, if_neg (by decide), innerCandidates21_14_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_15
    (if 3 ≤ innerCandidates21_14.length then search edge 3 [89, 1523] innerCandidates21_14 1612 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_15

def innerTail21_13 : List Nat := 1427 :: innerTail21_14
def innerCandidates21_13 : List Nat := [2777, 3989]
set_option maxRecDepth 4096 in
theorem innerCandidates21_13_checked : candidates edge 4 89 1427 innerTail21_14 best.bound = innerCandidates21_13 := by
  unfold candidates
  have row : edge 1427 = PrimePairCache.smallRow1427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_13 : scan edge (search edge 3) 4 [89] 89 innerTail21_13 best = best := by
  rw [innerTail21_13, scan, if_neg (by decide), innerCandidates21_13_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_14
    (if 3 ≤ innerCandidates21_13.length then search edge 3 [89, 1427] innerCandidates21_13 1516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_14

def innerTail21_12 : List Nat := 1409 :: innerTail21_13
def innerCandidates21_12 : List Nat := [5387, 6521]
set_option maxRecDepth 4096 in
theorem innerCandidates21_12_checked : candidates edge 4 89 1409 innerTail21_13 best.bound = innerCandidates21_12 := by
  unfold candidates
  have row : edge 1409 = PrimePairCache.smallRow1409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_12 : scan edge (search edge 3) 4 [89] 89 innerTail21_12 best = best := by
  rw [innerTail21_12, scan, if_neg (by decide), innerCandidates21_12_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_13
    (if 3 ≤ innerCandidates21_12.length then search edge 3 [89, 1409] innerCandidates21_12 1498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_13

def innerTail21_11 : List Nat := 1223 :: innerTail21_12
def innerCandidates21_11 : List Nat := [2141, 2579, 2777, 6113]
set_option maxRecDepth 4096 in
theorem innerCandidates21_11_checked : candidates edge 4 89 1223 innerTail21_12 best.bound = innerCandidates21_11 := by
  unfold candidates
  have row : edge 1223 = PrimePairCache.smallRow1223.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild21_11 : search edge 3 [89, 1223] innerCandidates21_11 1312 best = best := by decide

theorem innerScan21_11 : scan edge (search edge 3) 4 [89] 89 innerTail21_11 best = best := by
  rw [innerTail21_11, scan, if_neg (by decide), innerCandidates21_11_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_12
    (if 3 ≤ innerCandidates21_11.length then search edge 3 [89, 1223] innerCandidates21_11 1312 best else best) = best
  rw [if_pos (by decide), innerChild21_11]
  exact innerScan21_12

def innerTail21_10 : List Nat := 1061 :: innerTail21_11
def innerCandidates21_10 : List Nat := [4973]
set_option maxRecDepth 4096 in
theorem innerCandidates21_10_checked : candidates edge 4 89 1061 innerTail21_11 best.bound = innerCandidates21_10 := by
  unfold candidates
  have row : edge 1061 = PrimePairCache.smallRow1061.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_10 : scan edge (search edge 3) 4 [89] 89 innerTail21_10 best = best := by
  rw [innerTail21_10, scan, if_neg (by decide), innerCandidates21_10_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_11
    (if 3 ≤ innerCandidates21_10.length then search edge 3 [89, 1061] innerCandidates21_10 1150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_11

def innerTail21_9 : List Nat := 983 :: innerTail21_10
def innerCandidates21_9 : List Nat := [2267, 5171]
set_option maxRecDepth 4096 in
theorem innerCandidates21_9_checked : candidates edge 4 89 983 innerTail21_10 best.bound = innerCandidates21_9 := by
  unfold candidates
  have row : edge 983 = PrimePairCache.smallRow983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_9 : scan edge (search edge 3) 4 [89] 89 innerTail21_9 best = best := by
  rw [innerTail21_9, scan, if_neg (by decide), innerCandidates21_9_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_10
    (if 3 ≤ innerCandidates21_9.length then search edge 3 [89, 983] innerCandidates21_9 1072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_10

def innerTail21_8 : List Nat := 977 :: innerTail21_9
def innerCandidates21_8 : List Nat := [1409, 2141, 5039]
set_option maxRecDepth 4096 in
theorem innerCandidates21_8_checked : candidates edge 4 89 977 innerTail21_9 best.bound = innerCandidates21_8 := by
  unfold candidates
  have row : edge 977 = PrimePairCache.smallRow977.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild21_8 : search edge 3 [89, 977] innerCandidates21_8 1066 best = best := by decide

theorem innerScan21_8 : scan edge (search edge 3) 4 [89] 89 innerTail21_8 best = best := by
  rw [innerTail21_8, scan, if_neg (by decide), innerCandidates21_8_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_9
    (if 3 ≤ innerCandidates21_8.length then search edge 3 [89, 977] innerCandidates21_8 1066 best else best) = best
  rw [if_pos (by decide), innerChild21_8]
  exact innerScan21_9

def innerTail21_7 : List Nat := 821 :: innerTail21_8
def innerCandidates21_7 : List Nat := [6521]
set_option maxRecDepth 4096 in
theorem innerCandidates21_7_checked : candidates edge 4 89 821 innerTail21_8 best.bound = innerCandidates21_7 := by
  unfold candidates
  have row : edge 821 = PrimePairCache.smallRow821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_7 : scan edge (search edge 3) 4 [89] 89 innerTail21_7 best = best := by
  rw [innerTail21_7, scan, if_neg (by decide), innerCandidates21_7_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_8
    (if 3 ≤ innerCandidates21_7.length then search edge 3 [89, 821] innerCandidates21_7 910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_8

def innerTail21_6 : List Nat := 809 :: innerTail21_7
def innerCandidates21_6 : List Nat := [821, 983, 1427, 2141, 5393]
set_option maxRecDepth 4096 in
theorem innerCandidates21_6_checked : candidates edge 4 89 809 innerTail21_7 best.bound = innerCandidates21_6 := by
  unfold candidates
  have row : edge 809 = PrimePairCache.smallRow809.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild21_6 : search edge 3 [89, 809] innerCandidates21_6 898 best = best := by decide

theorem innerScan21_6 : scan edge (search edge 3) 4 [89] 89 innerTail21_6 best = best := by
  rw [innerTail21_6, scan, if_neg (by decide), innerCandidates21_6_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_7
    (if 3 ≤ innerCandidates21_6.length then search edge 3 [89, 809] innerCandidates21_6 898 best else best) = best
  rw [if_pos (by decide), innerChild21_6]
  exact innerScan21_7

def innerTail21_5 : List Nat := 521 :: innerTail21_6
def innerCandidates21_5 : List Nat := [809, 1523, 2777, 6353, 6521]
set_option maxRecDepth 4096 in
theorem innerCandidates21_5_checked : candidates edge 4 89 521 innerTail21_6 best.bound = innerCandidates21_5 := by
  unfold candidates
  have row : edge 521 = PrimePairCache.smallRow521.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild21_5 : search edge 3 [89, 521] innerCandidates21_5 610 best = best := by decide

theorem innerScan21_5 : scan edge (search edge 3) 4 [89] 89 innerTail21_5 best = best := by
  rw [innerTail21_5, scan, if_neg (by decide), innerCandidates21_5_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_6
    (if 3 ≤ innerCandidates21_5.length then search edge 3 [89, 521] innerCandidates21_5 610 best else best) = best
  rw [if_pos (by decide), innerChild21_5]
  exact innerScan21_6

def innerTail21_4 : List Nat := 443 :: innerTail21_5
def innerCandidates21_4 : List Nat := [983, 1523]
set_option maxRecDepth 4096 in
theorem innerCandidates21_4_checked : candidates edge 4 89 443 innerTail21_5 best.bound = innerCandidates21_4 := by
  unfold candidates
  have row : edge 443 = PrimePairCache.smallRow443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan21_4 : scan edge (search edge 3) 4 [89] 89 innerTail21_4 best = best := by
  rw [innerTail21_4, scan, if_neg (by decide), innerCandidates21_4_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_5
    (if 3 ≤ innerCandidates21_4.length then search edge 3 [89, 443] innerCandidates21_4 532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan21_5

def innerTail21_3 : List Nat := 431 :: innerTail21_4
def innerCandidates21_3 : List Nat := [983, 1823, 3671, 5903]
set_option maxRecDepth 4096 in
theorem innerCandidates21_3_checked : candidates edge 4 89 431 innerTail21_4 best.bound = innerCandidates21_3 := by
  unfold candidates
  have row : edge 431 = PrimePairCache.smallRow431.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild21_3 : search edge 3 [89, 431] innerCandidates21_3 520 best = best := by decide

theorem innerScan21_3 : scan edge (search edge 3) 4 [89] 89 innerTail21_3 best = best := by
  rw [innerTail21_3, scan, if_neg (by decide), innerCandidates21_3_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_4
    (if 3 ≤ innerCandidates21_3.length then search edge 3 [89, 431] innerCandidates21_3 520 best else best) = best
  rw [if_pos (by decide), innerChild21_3]
  exact innerScan21_4

def innerTail21_2 : List Nat := 293 :: innerTail21_3
def innerCandidates21_2 : List Nat := [1823, 2777, 5171, 5189]
set_option maxRecDepth 4096 in
theorem innerCandidates21_2_checked : candidates edge 4 89 293 innerTail21_3 best.bound = innerCandidates21_2 := by
  unfold candidates
  have row : edge 293 = PrimePairCache.smallRow293.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild21_2 : search edge 3 [89, 293] innerCandidates21_2 382 best = best := by decide

theorem innerScan21_2 : scan edge (search edge 3) 4 [89] 89 innerTail21_2 best = best := by
  rw [innerTail21_2, scan, if_neg (by decide), innerCandidates21_2_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_3
    (if 3 ≤ innerCandidates21_2.length then search edge 3 [89, 293] innerCandidates21_2 382 best else best) = best
  rw [if_pos (by decide), innerChild21_2]
  exact innerScan21_3

def innerTail21_1 : List Nat := 137 :: innerTail21_2
def innerCandidates21_1 : List Nat := [3989, 4721, 8147]
set_option maxRecDepth 4096 in
theorem innerCandidates21_1_checked : candidates edge 4 89 137 innerTail21_2 best.bound = innerCandidates21_1 := by
  unfold candidates
  have row : edge 137 = PrimePairCache.smallRow137.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild21_1 : search edge 3 [89, 137] innerCandidates21_1 226 best = best := by decide

theorem innerScan21_1 : scan edge (search edge 3) 4 [89] 89 innerTail21_1 best = best := by
  rw [innerTail21_1, scan, if_neg (by decide), innerCandidates21_1_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_2
    (if 3 ≤ innerCandidates21_1.length then search edge 3 [89, 137] innerCandidates21_1 226 best else best) = best
  rw [if_pos (by decide), innerChild21_1]
  exact innerScan21_2

def innerTail21_0 : List Nat := 107 :: innerTail21_1
def innerCandidates21_0 : List Nat := [1061, 4973, 5507]
set_option maxRecDepth 4096 in
theorem innerCandidates21_0_checked : candidates edge 4 89 107 innerTail21_1 best.bound = innerCandidates21_0 := by
  unfold candidates
  have row : edge 107 = PrimePairCache.smallRow107.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild21_0 : search edge 3 [89, 107] innerCandidates21_0 196 best = best := by decide

theorem innerScan21_0 : scan edge (search edge 3) 4 [89] 89 innerTail21_0 best = best := by
  rw [innerTail21_0, scan, if_neg (by decide), innerCandidates21_0_checked]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_1
    (if 3 ≤ innerCandidates21_0.length then search edge 3 [89, 107] innerCandidates21_0 196 best else best) = best
  rw [if_pos (by decide), innerChild21_0]
  exact innerScan21_1

theorem rootBranch21 : search edge 4 [89] rootCandidates21 89 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [89] 89 innerTail21_0 best = best
  exact innerScan21_0

end ProjectEulerSolutions.PrimePairUpper.Search
