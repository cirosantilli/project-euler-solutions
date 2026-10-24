import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates6 : List Nat := [47, 89, 311, 509, 677, 743, 761, 773, 827, 1031, 1109, 1223, 1289, 1481, 1493, 1607, 1613, 1871, 2003, 2129, 2333, 2357, 2591, 2633, 2699, 2741, 2753, 2819, 3083, 3407, 3557, 3617, 4007, 4139, 4211, 4463, 4547, 4721, 4931, 5009, 5051, 5441, 5483, 5591, 5813, 5849, 5927, 6053, 6323, 6329, 6449, 6563, 6659, 6917, 7151, 7331, 7487, 7547, 7607, 7673, 7691, 8009, 8093, 8171, 8237, 8291]
set_option maxRecDepth 4096 in
theorem rootCandidates6_checked : candidates edge 5 0 23 tail7 best.bound = rootCandidates6 := by
  unfold candidates
  have row : edge 23 = PrimePairCache.smallRow23.contains := by funext q; rfl
  rw [row]
  decide

def innerTail6_66 : List Nat := []
theorem innerScan6_66 : scan edge (search edge 3) 4 [23] 23 innerTail6_66 best = best := rfl

def innerTail6_65 : List Nat := 8291 :: innerTail6_66
def innerCandidates6_65 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates6_65_checked : candidates edge 4 23 8291 innerTail6_66 best.bound = innerCandidates6_65 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_65 : scan edge (search edge 3) 4 [23] 23 innerTail6_65 best = best := by
  rw [innerTail6_65, scan, if_neg (by decide), innerCandidates6_65_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_66
    (if 3 ≤ innerCandidates6_65.length then search edge 3 [23, 8291] innerCandidates6_65 8314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_66

def innerTail6_64 : List Nat := 8237 :: innerTail6_65
def innerCandidates6_64 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates6_64_checked : candidates edge 4 23 8237 innerTail6_65 best.bound = innerCandidates6_64 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_64 : scan edge (search edge 3) 4 [23] 23 innerTail6_64 best = best := by
  rw [innerTail6_64, scan, if_neg (by decide), innerCandidates6_64_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_65
    (if 3 ≤ innerCandidates6_64.length then search edge 3 [23, 8237] innerCandidates6_64 8260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_65

def innerTail6_63 : List Nat := 8171 :: innerTail6_64
def innerCandidates6_63 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates6_63_checked : candidates edge 4 23 8171 innerTail6_64 best.bound = innerCandidates6_63 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_63 : scan edge (search edge 3) 4 [23] 23 innerTail6_63 best = best := by
  rw [innerTail6_63, scan, if_neg (by decide), innerCandidates6_63_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_64
    (if 3 ≤ innerCandidates6_63.length then search edge 3 [23, 8171] innerCandidates6_63 8194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_64

def innerTail6_62 : List Nat := 8093 :: innerTail6_63
def innerCandidates6_62 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates6_62_checked : candidates edge 4 23 8093 innerTail6_63 best.bound = innerCandidates6_62 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_62 : scan edge (search edge 3) 4 [23] 23 innerTail6_62 best = best := by
  rw [innerTail6_62, scan, if_neg (by decide), innerCandidates6_62_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_63
    (if 3 ≤ innerCandidates6_62.length then search edge 3 [23, 8093] innerCandidates6_62 8116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_63

def innerTail6_61 : List Nat := 8009 :: innerTail6_62
def innerCandidates6_61 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates6_61_checked : candidates edge 4 23 8009 innerTail6_62 best.bound = innerCandidates6_61 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_61 : scan edge (search edge 3) 4 [23] 23 innerTail6_61 best = best := by
  rw [innerTail6_61, scan, if_neg (by decide), innerCandidates6_61_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_62
    (if 3 ≤ innerCandidates6_61.length then search edge 3 [23, 8009] innerCandidates6_61 8032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_62

def innerTail6_60 : List Nat := 7691 :: innerTail6_61
def innerCandidates6_60 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates6_60_checked : candidates edge 4 23 7691 innerTail6_61 best.bound = innerCandidates6_60 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_60 : scan edge (search edge 3) 4 [23] 23 innerTail6_60 best = best := by
  rw [innerTail6_60, scan, if_neg (by decide), innerCandidates6_60_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_61
    (if 3 ≤ innerCandidates6_60.length then search edge 3 [23, 7691] innerCandidates6_60 7714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_61

def innerTail6_59 : List Nat := 7673 :: innerTail6_60
def innerCandidates6_59 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates6_59_checked : candidates edge 4 23 7673 innerTail6_60 best.bound = innerCandidates6_59 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_59 : scan edge (search edge 3) 4 [23] 23 innerTail6_59 best = best := by
  rw [innerTail6_59, scan, if_neg (by decide), innerCandidates6_59_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_60
    (if 3 ≤ innerCandidates6_59.length then search edge 3 [23, 7673] innerCandidates6_59 7696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_60

def innerTail6_58 : List Nat := 7607 :: innerTail6_59
def innerCandidates6_58 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates6_58_checked : candidates edge 4 23 7607 innerTail6_59 best.bound = innerCandidates6_58 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_58 : scan edge (search edge 3) 4 [23] 23 innerTail6_58 best = best := by
  rw [innerTail6_58, scan, if_neg (by decide), innerCandidates6_58_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_59
    (if 3 ≤ innerCandidates6_58.length then search edge 3 [23, 7607] innerCandidates6_58 7630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_59

def innerTail6_57 : List Nat := 7547 :: innerTail6_58
def innerCandidates6_57 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates6_57_checked : candidates edge 4 23 7547 innerTail6_58 best.bound = innerCandidates6_57 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_57 : scan edge (search edge 3) 4 [23] 23 innerTail6_57 best = best := by
  rw [innerTail6_57, scan, if_neg (by decide), innerCandidates6_57_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_58
    (if 3 ≤ innerCandidates6_57.length then search edge 3 [23, 7547] innerCandidates6_57 7570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_58

def innerTail6_56 : List Nat := 7487 :: innerTail6_57
def innerCandidates6_56 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates6_56_checked : candidates edge 4 23 7487 innerTail6_57 best.bound = innerCandidates6_56 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_56 : scan edge (search edge 3) 4 [23] 23 innerTail6_56 best = best := by
  rw [innerTail6_56, scan, if_neg (by decide), innerCandidates6_56_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_57
    (if 3 ≤ innerCandidates6_56.length then search edge 3 [23, 7487] innerCandidates6_56 7510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_57

def innerTail6_55 : List Nat := 7331 :: innerTail6_56
def innerCandidates6_55 : List Nat := [7487]
set_option maxRecDepth 4096 in
theorem innerCandidates6_55_checked : candidates edge 4 23 7331 innerTail6_56 best.bound = innerCandidates6_55 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_55 : scan edge (search edge 3) 4 [23] 23 innerTail6_55 best = best := by
  rw [innerTail6_55, scan, if_neg (by decide), innerCandidates6_55_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_56
    (if 3 ≤ innerCandidates6_55.length then search edge 3 [23, 7331] innerCandidates6_55 7354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_56

def innerTail6_54 : List Nat := 7151 :: innerTail6_55
def innerCandidates6_54 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates6_54_checked : candidates edge 4 23 7151 innerTail6_55 best.bound = innerCandidates6_54 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_54 : scan edge (search edge 3) 4 [23] 23 innerTail6_54 best = best := by
  rw [innerTail6_54, scan, if_neg (by decide), innerCandidates6_54_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_55
    (if 3 ≤ innerCandidates6_54.length then search edge 3 [23, 7151] innerCandidates6_54 7174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_55

def innerTail6_53 : List Nat := 6917 :: innerTail6_54
def innerCandidates6_53 : List Nat := [7331]
set_option maxRecDepth 4096 in
theorem innerCandidates6_53_checked : candidates edge 4 23 6917 innerTail6_54 best.bound = innerCandidates6_53 := by
  unfold candidates
  have row : edge 6917 = SearchData.row6917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_53 : scan edge (search edge 3) 4 [23] 23 innerTail6_53 best = best := by
  rw [innerTail6_53, scan, if_neg (by decide), innerCandidates6_53_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_54
    (if 3 ≤ innerCandidates6_53.length then search edge 3 [23, 6917] innerCandidates6_53 6940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_54

def innerTail6_52 : List Nat := 6659 :: innerTail6_53
def innerCandidates6_52 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates6_52_checked : candidates edge 4 23 6659 innerTail6_53 best.bound = innerCandidates6_52 := by
  unfold candidates
  have row : edge 6659 = SearchData.row6659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_52 : scan edge (search edge 3) 4 [23] 23 innerTail6_52 best = best := by
  rw [innerTail6_52, scan, if_neg (by decide), innerCandidates6_52_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_53
    (if 3 ≤ innerCandidates6_52.length then search edge 3 [23, 6659] innerCandidates6_52 6682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_53

def innerTail6_51 : List Nat := 6563 :: innerTail6_52
def innerCandidates6_51 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates6_51_checked : candidates edge 4 23 6563 innerTail6_52 best.bound = innerCandidates6_51 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_51 : scan edge (search edge 3) 4 [23] 23 innerTail6_51 best = best := by
  rw [innerTail6_51, scan, if_neg (by decide), innerCandidates6_51_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_52
    (if 3 ≤ innerCandidates6_51.length then search edge 3 [23, 6563] innerCandidates6_51 6586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_52

def innerTail6_50 : List Nat := 6449 :: innerTail6_51
def innerCandidates6_50 : List Nat := [8171]
set_option maxRecDepth 4096 in
theorem innerCandidates6_50_checked : candidates edge 4 23 6449 innerTail6_51 best.bound = innerCandidates6_50 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_50 : scan edge (search edge 3) 4 [23] 23 innerTail6_50 best = best := by
  rw [innerTail6_50, scan, if_neg (by decide), innerCandidates6_50_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_51
    (if 3 ≤ innerCandidates6_50.length then search edge 3 [23, 6449] innerCandidates6_50 6472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_51

def innerTail6_49 : List Nat := 6329 :: innerTail6_50
def innerCandidates6_49 : List Nat := [7331]
set_option maxRecDepth 4096 in
theorem innerCandidates6_49_checked : candidates edge 4 23 6329 innerTail6_50 best.bound = innerCandidates6_49 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_49 : scan edge (search edge 3) 4 [23] 23 innerTail6_49 best = best := by
  rw [innerTail6_49, scan, if_neg (by decide), innerCandidates6_49_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_50
    (if 3 ≤ innerCandidates6_49.length then search edge 3 [23, 6329] innerCandidates6_49 6352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_50

def innerTail6_48 : List Nat := 6323 :: innerTail6_49
def innerCandidates6_48 : List Nat := [8237]
set_option maxRecDepth 4096 in
theorem innerCandidates6_48_checked : candidates edge 4 23 6323 innerTail6_49 best.bound = innerCandidates6_48 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_48 : scan edge (search edge 3) 4 [23] 23 innerTail6_48 best = best := by
  rw [innerTail6_48, scan, if_neg (by decide), innerCandidates6_48_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_49
    (if 3 ≤ innerCandidates6_48.length then search edge 3 [23, 6323] innerCandidates6_48 6346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_49

def innerTail6_47 : List Nat := 6053 :: innerTail6_48
def innerCandidates6_47 : List Nat := [8171]
set_option maxRecDepth 4096 in
theorem innerCandidates6_47_checked : candidates edge 4 23 6053 innerTail6_48 best.bound = innerCandidates6_47 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_47 : scan edge (search edge 3) 4 [23] 23 innerTail6_47 best = best := by
  rw [innerTail6_47, scan, if_neg (by decide), innerCandidates6_47_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_48
    (if 3 ≤ innerCandidates6_47.length then search edge 3 [23, 6053] innerCandidates6_47 6076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_48

def innerTail6_46 : List Nat := 5927 :: innerTail6_47
def innerCandidates6_46 : List Nat := [8237]
set_option maxRecDepth 4096 in
theorem innerCandidates6_46_checked : candidates edge 4 23 5927 innerTail6_47 best.bound = innerCandidates6_46 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_46 : scan edge (search edge 3) 4 [23] 23 innerTail6_46 best = best := by
  rw [innerTail6_46, scan, if_neg (by decide), innerCandidates6_46_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_47
    (if 3 ≤ innerCandidates6_46.length then search edge 3 [23, 5927] innerCandidates6_46 5950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_47

def innerTail6_45 : List Nat := 5849 :: innerTail6_46
def innerCandidates6_45 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates6_45_checked : candidates edge 4 23 5849 innerTail6_46 best.bound = innerCandidates6_45 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_45 : scan edge (search edge 3) 4 [23] 23 innerTail6_45 best = best := by
  rw [innerTail6_45, scan, if_neg (by decide), innerCandidates6_45_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_46
    (if 3 ≤ innerCandidates6_45.length then search edge 3 [23, 5849] innerCandidates6_45 5872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_46

def innerTail6_44 : List Nat := 5813 :: innerTail6_45
def innerCandidates6_44 : List Nat := [7151]
set_option maxRecDepth 4096 in
theorem innerCandidates6_44_checked : candidates edge 4 23 5813 innerTail6_45 best.bound = innerCandidates6_44 := by
  unfold candidates
  have row : edge 5813 = SearchData.row5813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_44 : scan edge (search edge 3) 4 [23] 23 innerTail6_44 best = best := by
  rw [innerTail6_44, scan, if_neg (by decide), innerCandidates6_44_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_45
    (if 3 ≤ innerCandidates6_44.length then search edge 3 [23, 5813] innerCandidates6_44 5836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_45

def innerTail6_43 : List Nat := 5591 :: innerTail6_44
def innerCandidates6_43 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates6_43_checked : candidates edge 4 23 5591 innerTail6_44 best.bound = innerCandidates6_43 := by
  unfold candidates
  have row : edge 5591 = SearchData.row5591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_43 : scan edge (search edge 3) 4 [23] 23 innerTail6_43 best = best := by
  rw [innerTail6_43, scan, if_neg (by decide), innerCandidates6_43_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_44
    (if 3 ≤ innerCandidates6_43.length then search edge 3 [23, 5591] innerCandidates6_43 5614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_44

def innerTail6_42 : List Nat := 5483 :: innerTail6_43
def innerCandidates6_42 : List Nat := [7487]
set_option maxRecDepth 4096 in
theorem innerCandidates6_42_checked : candidates edge 4 23 5483 innerTail6_43 best.bound = innerCandidates6_42 := by
  unfold candidates
  have row : edge 5483 = SearchData.row5483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_42 : scan edge (search edge 3) 4 [23] 23 innerTail6_42 best = best := by
  rw [innerTail6_42, scan, if_neg (by decide), innerCandidates6_42_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_43
    (if 3 ≤ innerCandidates6_42.length then search edge 3 [23, 5483] innerCandidates6_42 5506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_43

def innerTail6_41 : List Nat := 5441 :: innerTail6_42
def innerCandidates6_41 : List Nat := [5849, 7547]
set_option maxRecDepth 4096 in
theorem innerCandidates6_41_checked : candidates edge 4 23 5441 innerTail6_42 best.bound = innerCandidates6_41 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_41 : scan edge (search edge 3) 4 [23] 23 innerTail6_41 best = best := by
  rw [innerTail6_41, scan, if_neg (by decide), innerCandidates6_41_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_42
    (if 3 ≤ innerCandidates6_41.length then search edge 3 [23, 5441] innerCandidates6_41 5464 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_42

def innerTail6_40 : List Nat := 5051 :: innerTail6_41
def innerCandidates6_40 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates6_40_checked : candidates edge 4 23 5051 innerTail6_41 best.bound = innerCandidates6_40 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_40 : scan edge (search edge 3) 4 [23] 23 innerTail6_40 best = best := by
  rw [innerTail6_40, scan, if_neg (by decide), innerCandidates6_40_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_41
    (if 3 ≤ innerCandidates6_40.length then search edge 3 [23, 5051] innerCandidates6_40 5074 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_41

def innerTail6_39 : List Nat := 5009 :: innerTail6_40
def innerCandidates6_39 : List Nat := [5849, 8093]
set_option maxRecDepth 4096 in
theorem innerCandidates6_39_checked : candidates edge 4 23 5009 innerTail6_40 best.bound = innerCandidates6_39 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_39 : scan edge (search edge 3) 4 [23] 23 innerTail6_39 best = best := by
  rw [innerTail6_39, scan, if_neg (by decide), innerCandidates6_39_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_40
    (if 3 ≤ innerCandidates6_39.length then search edge 3 [23, 5009] innerCandidates6_39 5032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_40

def innerTail6_38 : List Nat := 4931 :: innerTail6_39
def innerCandidates6_38 : List Nat := [5009]
set_option maxRecDepth 4096 in
theorem innerCandidates6_38_checked : candidates edge 4 23 4931 innerTail6_39 best.bound = innerCandidates6_38 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_38 : scan edge (search edge 3) 4 [23] 23 innerTail6_38 best = best := by
  rw [innerTail6_38, scan, if_neg (by decide), innerCandidates6_38_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_39
    (if 3 ≤ innerCandidates6_38.length then search edge 3 [23, 4931] innerCandidates6_38 4954 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_39

def innerTail6_37 : List Nat := 4721 :: innerTail6_38
def innerCandidates6_37 : List Nat := [5813]
set_option maxRecDepth 4096 in
theorem innerCandidates6_37_checked : candidates edge 4 23 4721 innerTail6_38 best.bound = innerCandidates6_37 := by
  unfold candidates
  have row : edge 4721 = SearchData.row4721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_37 : scan edge (search edge 3) 4 [23] 23 innerTail6_37 best = best := by
  rw [innerTail6_37, scan, if_neg (by decide), innerCandidates6_37_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_38
    (if 3 ≤ innerCandidates6_37.length then search edge 3 [23, 4721] innerCandidates6_37 4744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_38

def innerTail6_36 : List Nat := 4547 :: innerTail6_37
def innerCandidates6_36 : List Nat := [7547]
set_option maxRecDepth 4096 in
theorem innerCandidates6_36_checked : candidates edge 4 23 4547 innerTail6_37 best.bound = innerCandidates6_36 := by
  unfold candidates
  have row : edge 4547 = SearchData.row4547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_36 : scan edge (search edge 3) 4 [23] 23 innerTail6_36 best = best := by
  rw [innerTail6_36, scan, if_neg (by decide), innerCandidates6_36_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_37
    (if 3 ≤ innerCandidates6_36.length then search edge 3 [23, 4547] innerCandidates6_36 4570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_37

def innerTail6_35 : List Nat := 4463 :: innerTail6_36
def innerCandidates6_35 : List Nat := [5441, 5813]
set_option maxRecDepth 4096 in
theorem innerCandidates6_35_checked : candidates edge 4 23 4463 innerTail6_36 best.bound = innerCandidates6_35 := by
  unfold candidates
  have row : edge 4463 = SearchData.row4463.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_35 : scan edge (search edge 3) 4 [23] 23 innerTail6_35 best = best := by
  rw [innerTail6_35, scan, if_neg (by decide), innerCandidates6_35_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_36
    (if 3 ≤ innerCandidates6_35.length then search edge 3 [23, 4463] innerCandidates6_35 4486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_36

def innerTail6_34 : List Nat := 4211 :: innerTail6_35
def innerCandidates6_34 : List Nat := [5483]
set_option maxRecDepth 4096 in
theorem innerCandidates6_34_checked : candidates edge 4 23 4211 innerTail6_35 best.bound = innerCandidates6_34 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_34 : scan edge (search edge 3) 4 [23] 23 innerTail6_34 best = best := by
  rw [innerTail6_34, scan, if_neg (by decide), innerCandidates6_34_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_35
    (if 3 ≤ innerCandidates6_34.length then search edge 3 [23, 4211] innerCandidates6_34 4234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_35

def innerTail6_33 : List Nat := 4139 :: innerTail6_34
def innerCandidates6_33 : List Nat := [7607, 7673]
set_option maxRecDepth 4096 in
theorem innerCandidates6_33_checked : candidates edge 4 23 4139 innerTail6_34 best.bound = innerCandidates6_33 := by
  unfold candidates
  have row : edge 4139 = SearchData.row4139.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_33 : scan edge (search edge 3) 4 [23] 23 innerTail6_33 best = best := by
  rw [innerTail6_33, scan, if_neg (by decide), innerCandidates6_33_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_34
    (if 3 ≤ innerCandidates6_33.length then search edge 3 [23, 4139] innerCandidates6_33 4162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_34

def innerTail6_32 : List Nat := 4007 :: innerTail6_33
def innerCandidates6_32 : List Nat := [6053]
set_option maxRecDepth 4096 in
theorem innerCandidates6_32_checked : candidates edge 4 23 4007 innerTail6_33 best.bound = innerCandidates6_32 := by
  unfold candidates
  have row : edge 4007 = SearchData.row4007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_32 : scan edge (search edge 3) 4 [23] 23 innerTail6_32 best = best := by
  rw [innerTail6_32, scan, if_neg (by decide), innerCandidates6_32_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_33
    (if 3 ≤ innerCandidates6_32.length then search edge 3 [23, 4007] innerCandidates6_32 4030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_33

def innerTail6_31 : List Nat := 3617 :: innerTail6_32
def innerCandidates6_31 : List Nat := [7607, 8291]
set_option maxRecDepth 4096 in
theorem innerCandidates6_31_checked : candidates edge 4 23 3617 innerTail6_32 best.bound = innerCandidates6_31 := by
  unfold candidates
  have row : edge 3617 = SearchData.row3617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_31 : scan edge (search edge 3) 4 [23] 23 innerTail6_31 best = best := by
  rw [innerTail6_31, scan, if_neg (by decide), innerCandidates6_31_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_32
    (if 3 ≤ innerCandidates6_31.length then search edge 3 [23, 3617] innerCandidates6_31 3640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_32

def innerTail6_30 : List Nat := 3557 :: innerTail6_31
def innerCandidates6_30 : List Nat := [4139, 5051, 6329]
set_option maxRecDepth 4096 in
theorem innerCandidates6_30_checked : candidates edge 4 23 3557 innerTail6_31 best.bound = innerCandidates6_30 := by
  unfold candidates
  have row : edge 3557 = SearchData.row3557.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild6_30 : search edge 3 [23, 3557] innerCandidates6_30 3580 best = best := by decide

theorem innerScan6_30 : scan edge (search edge 3) 4 [23] 23 innerTail6_30 best = best := by
  rw [innerTail6_30, scan, if_neg (by decide), innerCandidates6_30_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_31
    (if 3 ≤ innerCandidates6_30.length then search edge 3 [23, 3557] innerCandidates6_30 3580 best else best) = best
  rw [if_pos (by decide), innerChild6_30]
  exact innerScan6_31

def innerTail6_29 : List Nat := 3407 :: innerTail6_30
def innerCandidates6_29 : List Nat := [7691]
set_option maxRecDepth 4096 in
theorem innerCandidates6_29_checked : candidates edge 4 23 3407 innerTail6_30 best.bound = innerCandidates6_29 := by
  unfold candidates
  have row : edge 3407 = SearchData.row3407.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_29 : scan edge (search edge 3) 4 [23] 23 innerTail6_29 best = best := by
  rw [innerTail6_29, scan, if_neg (by decide), innerCandidates6_29_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_30
    (if 3 ≤ innerCandidates6_29.length then search edge 3 [23, 3407] innerCandidates6_29 3430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_30

def innerTail6_28 : List Nat := 3083 :: innerTail6_29
def innerCandidates6_28 : List Nat := [4931]
set_option maxRecDepth 4096 in
theorem innerCandidates6_28_checked : candidates edge 4 23 3083 innerTail6_29 best.bound = innerCandidates6_28 := by
  unfold candidates
  have row : edge 3083 = SearchData.row3083.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_28 : scan edge (search edge 3) 4 [23] 23 innerTail6_28 best = best := by
  rw [innerTail6_28, scan, if_neg (by decide), innerCandidates6_28_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_29
    (if 3 ≤ innerCandidates6_28.length then search edge 3 [23, 3083] innerCandidates6_28 3106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_29

def innerTail6_27 : List Nat := 2819 :: innerTail6_28
def innerCandidates6_27 : List Nat := [3617, 5927, 7607]
set_option maxRecDepth 4096 in
theorem innerCandidates6_27_checked : candidates edge 4 23 2819 innerTail6_28 best.bound = innerCandidates6_27 := by
  unfold candidates
  have row : edge 2819 = SearchData.row2819.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild6_27 : search edge 3 [23, 2819] innerCandidates6_27 2842 best = best := by decide

theorem innerScan6_27 : scan edge (search edge 3) 4 [23] 23 innerTail6_27 best = best := by
  rw [innerTail6_27, scan, if_neg (by decide), innerCandidates6_27_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_28
    (if 3 ≤ innerCandidates6_27.length then search edge 3 [23, 2819] innerCandidates6_27 2842 best else best) = best
  rw [if_pos (by decide), innerChild6_27]
  exact innerScan6_28

def innerTail6_26 : List Nat := 2753 :: innerTail6_27
def innerCandidates6_26 : List Nat := [7487]
set_option maxRecDepth 4096 in
theorem innerCandidates6_26_checked : candidates edge 4 23 2753 innerTail6_27 best.bound = innerCandidates6_26 := by
  unfold candidates
  have row : edge 2753 = SearchData.row2753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_26 : scan edge (search edge 3) 4 [23] 23 innerTail6_26 best = best := by
  rw [innerTail6_26, scan, if_neg (by decide), innerCandidates6_26_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_27
    (if 3 ≤ innerCandidates6_26.length then search edge 3 [23, 2753] innerCandidates6_26 2776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_27

def innerTail6_25 : List Nat := 2741 :: innerTail6_26
def innerCandidates6_25 : List Nat := [7331]
set_option maxRecDepth 4096 in
theorem innerCandidates6_25_checked : candidates edge 4 23 2741 innerTail6_26 best.bound = innerCandidates6_25 := by
  unfold candidates
  have row : edge 2741 = SearchData.row2741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_25 : scan edge (search edge 3) 4 [23] 23 innerTail6_25 best = best := by
  rw [innerTail6_25, scan, if_neg (by decide), innerCandidates6_25_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_26
    (if 3 ≤ innerCandidates6_25.length then search edge 3 [23, 2741] innerCandidates6_25 2764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_26

def innerTail6_24 : List Nat := 2699 :: innerTail6_25
def innerCandidates6_24 : List Nat := [6329]
set_option maxRecDepth 4096 in
theorem innerCandidates6_24_checked : candidates edge 4 23 2699 innerTail6_25 best.bound = innerCandidates6_24 := by
  unfold candidates
  have row : edge 2699 = SearchData.row2699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_24 : scan edge (search edge 3) 4 [23] 23 innerTail6_24 best = best := by
  rw [innerTail6_24, scan, if_neg (by decide), innerCandidates6_24_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_25
    (if 3 ≤ innerCandidates6_24.length then search edge 3 [23, 2699] innerCandidates6_24 2722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_25

def innerTail6_23 : List Nat := 2633 :: innerTail6_24
def innerCandidates6_23 : List Nat := [2699, 5051, 7547]
set_option maxRecDepth 4096 in
theorem innerCandidates6_23_checked : candidates edge 4 23 2633 innerTail6_24 best.bound = innerCandidates6_23 := by
  unfold candidates
  have row : edge 2633 = PrimePairCache.smallRow2633.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild6_23 : search edge 3 [23, 2633] innerCandidates6_23 2656 best = best := by decide

theorem innerScan6_23 : scan edge (search edge 3) 4 [23] 23 innerTail6_23 best = best := by
  rw [innerTail6_23, scan, if_neg (by decide), innerCandidates6_23_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_24
    (if 3 ≤ innerCandidates6_23.length then search edge 3 [23, 2633] innerCandidates6_23 2656 best else best) = best
  rw [if_pos (by decide), innerChild6_23]
  exact innerScan6_24

def innerTail6_22 : List Nat := 2591 :: innerTail6_23
def innerCandidates6_22 : List Nat := [2633, 4211, 5849]
set_option maxRecDepth 4096 in
theorem innerCandidates6_22_checked : candidates edge 4 23 2591 innerTail6_23 best.bound = innerCandidates6_22 := by
  unfold candidates
  have row : edge 2591 = PrimePairCache.smallRow2591.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild6_22 : search edge 3 [23, 2591] innerCandidates6_22 2614 best = best := by decide

theorem innerScan6_22 : scan edge (search edge 3) 4 [23] 23 innerTail6_22 best = best := by
  rw [innerTail6_22, scan, if_neg (by decide), innerCandidates6_22_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_23
    (if 3 ≤ innerCandidates6_22.length then search edge 3 [23, 2591] innerCandidates6_22 2614 best else best) = best
  rw [if_pos (by decide), innerChild6_22]
  exact innerScan6_23

def innerTail6_21 : List Nat := 2357 :: innerTail6_22
def innerCandidates6_21 : List Nat := [6323, 7331, 8237]
set_option maxRecDepth 4096 in
theorem innerCandidates6_21_checked : candidates edge 4 23 2357 innerTail6_22 best.bound = innerCandidates6_21 := by
  unfold candidates
  have row : edge 2357 = PrimePairCache.smallRow2357.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild6_21 : search edge 3 [23, 2357] innerCandidates6_21 2380 best = best := by decide

theorem innerScan6_21 : scan edge (search edge 3) 4 [23] 23 innerTail6_21 best = best := by
  rw [innerTail6_21, scan, if_neg (by decide), innerCandidates6_21_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_22
    (if 3 ≤ innerCandidates6_21.length then search edge 3 [23, 2357] innerCandidates6_21 2380 best else best) = best
  rw [if_pos (by decide), innerChild6_21]
  exact innerScan6_22

def innerTail6_20 : List Nat := 2333 :: innerTail6_21
def innerCandidates6_20 : List Nat := [5927]
set_option maxRecDepth 4096 in
theorem innerCandidates6_20_checked : candidates edge 4 23 2333 innerTail6_21 best.bound = innerCandidates6_20 := by
  unfold candidates
  have row : edge 2333 = PrimePairCache.smallRow2333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_20 : scan edge (search edge 3) 4 [23] 23 innerTail6_20 best = best := by
  rw [innerTail6_20, scan, if_neg (by decide), innerCandidates6_20_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_21
    (if 3 ≤ innerCandidates6_20.length then search edge 3 [23, 2333] innerCandidates6_20 2356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_21

def innerTail6_19 : List Nat := 2129 :: innerTail6_20
def innerCandidates6_19 : List Nat := [5051]
set_option maxRecDepth 4096 in
theorem innerCandidates6_19_checked : candidates edge 4 23 2129 innerTail6_20 best.bound = innerCandidates6_19 := by
  unfold candidates
  have row : edge 2129 = PrimePairCache.smallRow2129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_19 : scan edge (search edge 3) 4 [23] 23 innerTail6_19 best = best := by
  rw [innerTail6_19, scan, if_neg (by decide), innerCandidates6_19_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_20
    (if 3 ≤ innerCandidates6_19.length then search edge 3 [23, 2129] innerCandidates6_19 2152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_20

def innerTail6_18 : List Nat := 2003 :: innerTail6_19
def innerCandidates6_18 : List Nat := [3557, 5051, 6563, 8171]
set_option maxRecDepth 4096 in
theorem innerCandidates6_18_checked : candidates edge 4 23 2003 innerTail6_19 best.bound = innerCandidates6_18 := by
  unfold candidates
  have row : edge 2003 = PrimePairCache.smallRow2003.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild6_18 : search edge 3 [23, 2003] innerCandidates6_18 2026 best = best := by decide

theorem innerScan6_18 : scan edge (search edge 3) 4 [23] 23 innerTail6_18 best = best := by
  rw [innerTail6_18, scan, if_neg (by decide), innerCandidates6_18_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_19
    (if 3 ≤ innerCandidates6_18.length then search edge 3 [23, 2003] innerCandidates6_18 2026 best else best) = best
  rw [if_pos (by decide), innerChild6_18]
  exact innerScan6_19

def innerTail6_17 : List Nat := 1871 :: innerTail6_18
def innerCandidates6_17 : List Nat := [4139, 5051, 5441, 6323, 8171]
set_option maxRecDepth 4096 in
theorem innerCandidates6_17_checked : candidates edge 4 23 1871 innerTail6_18 best.bound = innerCandidates6_17 := by
  unfold candidates
  have row : edge 1871 = PrimePairCache.smallRow1871.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild6_17 : search edge 3 [23, 1871] innerCandidates6_17 1894 best = best := by decide

theorem innerScan6_17 : scan edge (search edge 3) 4 [23] 23 innerTail6_17 best = best := by
  rw [innerTail6_17, scan, if_neg (by decide), innerCandidates6_17_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_18
    (if 3 ≤ innerCandidates6_17.length then search edge 3 [23, 1871] innerCandidates6_17 1894 best else best) = best
  rw [if_pos (by decide), innerChild6_17]
  exact innerScan6_18

def innerTail6_16 : List Nat := 1613 :: innerTail6_17
def innerCandidates6_16 : List Nat := [5813, 7487]
set_option maxRecDepth 4096 in
theorem innerCandidates6_16_checked : candidates edge 4 23 1613 innerTail6_17 best.bound = innerCandidates6_16 := by
  unfold candidates
  have row : edge 1613 = PrimePairCache.smallRow1613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_16 : scan edge (search edge 3) 4 [23] 23 innerTail6_16 best = best := by
  rw [innerTail6_16, scan, if_neg (by decide), innerCandidates6_16_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_17
    (if 3 ≤ innerCandidates6_16.length then search edge 3 [23, 1613] innerCandidates6_16 1636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_17

def innerTail6_15 : List Nat := 1607 :: innerTail6_16
def innerCandidates6_15 : List Nat := [5009, 5849, 7673, 8171]
set_option maxRecDepth 4096 in
theorem innerCandidates6_15_checked : candidates edge 4 23 1607 innerTail6_16 best.bound = innerCandidates6_15 := by
  unfold candidates
  have row : edge 1607 = PrimePairCache.smallRow1607.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild6_15 : search edge 3 [23, 1607] innerCandidates6_15 1630 best = best := by decide

theorem innerScan6_15 : scan edge (search edge 3) 4 [23] 23 innerTail6_15 best = best := by
  rw [innerTail6_15, scan, if_neg (by decide), innerCandidates6_15_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_16
    (if 3 ≤ innerCandidates6_15.length then search edge 3 [23, 1607] innerCandidates6_15 1630 best else best) = best
  rw [if_pos (by decide), innerChild6_15]
  exact innerScan6_16

def innerTail6_14 : List Nat := 1493 :: innerTail6_15
def innerCandidates6_14 : List Nat := [7607]
set_option maxRecDepth 4096 in
theorem innerCandidates6_14_checked : candidates edge 4 23 1493 innerTail6_15 best.bound = innerCandidates6_14 := by
  unfold candidates
  have row : edge 1493 = PrimePairCache.smallRow1493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_14 : scan edge (search edge 3) 4 [23] 23 innerTail6_14 best = best := by
  rw [innerTail6_14, scan, if_neg (by decide), innerCandidates6_14_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_15
    (if 3 ≤ innerCandidates6_14.length then search edge 3 [23, 1493] innerCandidates6_14 1516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_15

def innerTail6_13 : List Nat := 1481 :: innerTail6_14
def innerCandidates6_13 : List Nat := [4211]
set_option maxRecDepth 4096 in
theorem innerCandidates6_13_checked : candidates edge 4 23 1481 innerTail6_14 best.bound = innerCandidates6_13 := by
  unfold candidates
  have row : edge 1481 = PrimePairCache.smallRow1481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_13 : scan edge (search edge 3) 4 [23] 23 innerTail6_13 best = best := by
  rw [innerTail6_13, scan, if_neg (by decide), innerCandidates6_13_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_14
    (if 3 ≤ innerCandidates6_13.length then search edge 3 [23, 1481] innerCandidates6_13 1504 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_14

def innerTail6_12 : List Nat := 1289 :: innerTail6_13
def innerCandidates6_12 : List Nat := [4139, 6449]
set_option maxRecDepth 4096 in
theorem innerCandidates6_12_checked : candidates edge 4 23 1289 innerTail6_13 best.bound = innerCandidates6_12 := by
  unfold candidates
  have row : edge 1289 = PrimePairCache.smallRow1289.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_12 : scan edge (search edge 3) 4 [23] 23 innerTail6_12 best = best := by
  rw [innerTail6_12, scan, if_neg (by decide), innerCandidates6_12_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_13
    (if 3 ≤ innerCandidates6_12.length then search edge 3 [23, 1289] innerCandidates6_12 1312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_13

def innerTail6_11 : List Nat := 1223 :: innerTail6_12
def innerCandidates6_11 : List Nat := [1493]
set_option maxRecDepth 4096 in
theorem innerCandidates6_11_checked : candidates edge 4 23 1223 innerTail6_12 best.bound = innerCandidates6_11 := by
  unfold candidates
  have row : edge 1223 = PrimePairCache.smallRow1223.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_11 : scan edge (search edge 3) 4 [23] 23 innerTail6_11 best = best := by
  rw [innerTail6_11, scan, if_neg (by decide), innerCandidates6_11_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_12
    (if 3 ≤ innerCandidates6_11.length then search edge 3 [23, 1223] innerCandidates6_11 1246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_12

def innerTail6_10 : List Nat := 1109 :: innerTail6_11
def innerCandidates6_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates6_10_checked : candidates edge 4 23 1109 innerTail6_11 best.bound = innerCandidates6_10 := by
  unfold candidates
  have row : edge 1109 = PrimePairCache.smallRow1109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_10 : scan edge (search edge 3) 4 [23] 23 innerTail6_10 best = best := by
  rw [innerTail6_10, scan, if_neg (by decide), innerCandidates6_10_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_11
    (if 3 ≤ innerCandidates6_10.length then search edge 3 [23, 1109] innerCandidates6_10 1132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_11

def innerTail6_9 : List Nat := 1031 :: innerTail6_10
def innerCandidates6_9 : List Nat := [1481]
set_option maxRecDepth 4096 in
theorem innerCandidates6_9_checked : candidates edge 4 23 1031 innerTail6_10 best.bound = innerCandidates6_9 := by
  unfold candidates
  have row : edge 1031 = PrimePairCache.smallRow1031.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan6_9 : scan edge (search edge 3) 4 [23] 23 innerTail6_9 best = best := by
  rw [innerTail6_9, scan, if_neg (by decide), innerCandidates6_9_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_10
    (if 3 ≤ innerCandidates6_9.length then search edge 3 [23, 1031] innerCandidates6_9 1054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan6_10

def innerTail6_8 : List Nat := 827 :: innerTail6_9
def innerCandidates6_8 : List Nat := [1871, 2819, 4463]
set_option maxRecDepth 4096 in
theorem innerCandidates6_8_checked : candidates edge 4 23 827 innerTail6_9 best.bound = innerCandidates6_8 := by
  unfold candidates
  have row : edge 827 = PrimePairCache.smallRow827.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild6_8 : search edge 3 [23, 827] innerCandidates6_8 850 best = best := by decide

theorem innerScan6_8 : scan edge (search edge 3) 4 [23] 23 innerTail6_8 best = best := by
  rw [innerTail6_8, scan, if_neg (by decide), innerCandidates6_8_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_9
    (if 3 ≤ innerCandidates6_8.length then search edge 3 [23, 827] innerCandidates6_8 850 best else best) = best
  rw [if_pos (by decide), innerChild6_8]
  exact innerScan6_9

def innerTail6_7 : List Nat := 773 :: innerTail6_8
def innerCandidates6_7 : List Nat := [2129, 2591, 5927, 7547]
set_option maxRecDepth 4096 in
theorem innerCandidates6_7_checked : candidates edge 4 23 773 innerTail6_8 best.bound = innerCandidates6_7 := by
  unfold candidates
  have row : edge 773 = PrimePairCache.smallRow773.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild6_7 : search edge 3 [23, 773] innerCandidates6_7 796 best = best := by decide

theorem innerScan6_7 : scan edge (search edge 3) 4 [23] 23 innerTail6_7 best = best := by
  rw [innerTail6_7, scan, if_neg (by decide), innerCandidates6_7_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_8
    (if 3 ≤ innerCandidates6_7.length then search edge 3 [23, 773] innerCandidates6_7 796 best else best) = best
  rw [if_pos (by decide), innerChild6_7]
  exact innerScan6_8

def innerTail6_6 : List Nat := 761 :: innerTail6_7
def innerCandidates6_6 : List Nat := [1613, 4463, 5009]
set_option maxRecDepth 4096 in
theorem innerCandidates6_6_checked : candidates edge 4 23 761 innerTail6_7 best.bound = innerCandidates6_6 := by
  unfold candidates
  have row : edge 761 = PrimePairCache.smallRow761.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild6_6 : search edge 3 [23, 761] innerCandidates6_6 784 best = best := by decide

theorem innerScan6_6 : scan edge (search edge 3) 4 [23] 23 innerTail6_6 best = best := by
  rw [innerTail6_6, scan, if_neg (by decide), innerCandidates6_6_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_7
    (if 3 ≤ innerCandidates6_6.length then search edge 3 [23, 761] innerCandidates6_6 784 best else best) = best
  rw [if_pos (by decide), innerChild6_6]
  exact innerScan6_7

def innerTail6_5 : List Nat := 743 :: innerTail6_6
def innerCandidates6_5 : List Nat := [1481, 1493, 1871, 2753, 5591, 6323]
set_option maxRecDepth 4096 in
theorem innerCandidates6_5_checked : candidates edge 4 23 743 innerTail6_6 best.bound = innerCandidates6_5 := by
  unfold candidates
  have row : edge 743 = PrimePairCache.smallRow743.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild6_5 : search edge 3 [23, 743] innerCandidates6_5 766 best = best := by decide

theorem innerScan6_5 : scan edge (search edge 3) 4 [23] 23 innerTail6_5 best = best := by
  rw [innerTail6_5, scan, if_neg (by decide), innerCandidates6_5_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_6
    (if 3 ≤ innerCandidates6_5.length then search edge 3 [23, 743] innerCandidates6_5 766 best else best) = best
  rw [if_pos (by decide), innerChild6_5]
  exact innerScan6_6

def innerTail6_4 : List Nat := 677 :: innerTail6_5
def innerCandidates6_4 : List Nat := [827, 1493, 1871]
set_option maxRecDepth 4096 in
theorem innerCandidates6_4_checked : candidates edge 4 23 677 innerTail6_5 best.bound = innerCandidates6_4 := by
  unfold candidates
  have row : edge 677 = PrimePairCache.smallRow677.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild6_4 : search edge 3 [23, 677] innerCandidates6_4 700 best = best := by decide

theorem innerScan6_4 : scan edge (search edge 3) 4 [23] 23 innerTail6_4 best = best := by
  rw [innerTail6_4, scan, if_neg (by decide), innerCandidates6_4_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_5
    (if 3 ≤ innerCandidates6_4.length then search edge 3 [23, 677] innerCandidates6_4 700 best else best) = best
  rw [if_pos (by decide), innerChild6_4]
  exact innerScan6_5

def innerTail6_3 : List Nat := 509 :: innerTail6_4
def innerCandidates6_3 : List Nat := [1613, 5483, 6563]
set_option maxRecDepth 4096 in
theorem innerCandidates6_3_checked : candidates edge 4 23 509 innerTail6_4 best.bound = innerCandidates6_3 := by
  unfold candidates
  have row : edge 509 = PrimePairCache.smallRow509.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild6_3 : search edge 3 [23, 509] innerCandidates6_3 532 best = best := by decide

theorem innerScan6_3 : scan edge (search edge 3) 4 [23] 23 innerTail6_3 best = best := by
  rw [innerTail6_3, scan, if_neg (by decide), innerCandidates6_3_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_4
    (if 3 ≤ innerCandidates6_3.length then search edge 3 [23, 509] innerCandidates6_3 532 best else best) = best
  rw [if_pos (by decide), innerChild6_3]
  exact innerScan6_4

def innerTail6_2 : List Nat := 311 :: innerTail6_3
def innerCandidates6_2 : List Nat := [677, 827, 1223, 8237]
set_option maxRecDepth 4096 in
theorem innerCandidates6_2_checked : candidates edge 4 23 311 innerTail6_3 best.bound = innerCandidates6_2 := by
  unfold candidates
  have row : edge 311 = PrimePairCache.smallRow311.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild6_2 : search edge 3 [23, 311] innerCandidates6_2 334 best = best := by decide

theorem innerScan6_2 : scan edge (search edge 3) 4 [23] 23 innerTail6_2 best = best := by
  rw [innerTail6_2, scan, if_neg (by decide), innerCandidates6_2_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_3
    (if 3 ≤ innerCandidates6_2.length then search edge 3 [23, 311] innerCandidates6_2 334 best else best) = best
  rw [if_pos (by decide), innerChild6_2]
  exact innerScan6_3

def innerTail6_1 : List Nat := 89 :: innerTail6_2
def innerCandidates6_1 : List Nat := [1223, 2357, 3407, 4211, 4721, 6323]
set_option maxRecDepth 4096 in
theorem innerCandidates6_1_checked : candidates edge 4 23 89 innerTail6_2 best.bound = innerCandidates6_1 := by
  unfold candidates
  have row : edge 89 = PrimePairCache.smallRow89.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild6_1 : search edge 3 [23, 89] innerCandidates6_1 112 best = best := by decide

theorem innerScan6_1 : scan edge (search edge 3) 4 [23] 23 innerTail6_1 best = best := by
  rw [innerTail6_1, scan, if_neg (by decide), innerCandidates6_1_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_2
    (if 3 ≤ innerCandidates6_1.length then search edge 3 [23, 89] innerCandidates6_1 112 best else best) = best
  rw [if_pos (by decide), innerChild6_1]
  exact innerScan6_2

def innerTail6_0 : List Nat := 47 :: innerTail6_1
def innerCandidates6_0 : List Nat := [1481, 2333, 2741, 3617, 4211, 4931, 5483, 5927, 6659, 8171]
set_option maxRecDepth 4096 in
theorem innerCandidates6_0_checked : candidates edge 4 23 47 innerTail6_1 best.bound = innerCandidates6_0 := by
  unfold candidates
  have row : edge 47 = PrimePairCache.smallRow47.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild6_0 : search edge 3 [23, 47] innerCandidates6_0 70 best = best := by decide

theorem innerScan6_0 : scan edge (search edge 3) 4 [23] 23 innerTail6_0 best = best := by
  rw [innerTail6_0, scan, if_neg (by decide), innerCandidates6_0_checked]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_1
    (if 3 ≤ innerCandidates6_0.length then search edge 3 [23, 47] innerCandidates6_0 70 best else best) = best
  rw [if_pos (by decide), innerChild6_0]
  exact innerScan6_1

theorem rootBranch6 : search edge 4 [23] rootCandidates6 23 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [23] 23 innerTail6_0 best = best
  exact innerScan6_0

end ProjectEulerSolutions.PrimePairUpper.Search
