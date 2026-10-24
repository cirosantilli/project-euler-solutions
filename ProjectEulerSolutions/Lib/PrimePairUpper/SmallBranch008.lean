import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates983 : List Nat := [7867, 7993]
set_option maxRecDepth 4096 in
theorem rootCandidates983_checked : candidates edge 5 0 7789 tail984 best.bound = rootCandidates983 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates982 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates982_checked : candidates edge 5 0 7759 tail983 best.bound = rootCandidates982 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates981 : List Nat := [7853, 7883, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates981_checked : candidates edge 5 0 7757 tail982 best.bound = rootCandidates981 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates980 : List Nat := [7927, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates980_checked : candidates edge 5 0 7753 tail981 best.bound = rootCandidates980 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates979 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates979_checked : candidates edge 5 0 7741 tail980 best.bound = rootCandidates979 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates978 : List Nat := [7853, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates978_checked : candidates edge 5 0 7727 tail979 best.bound = rootCandidates978 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates977 : List Nat := []
set_option maxRecDepth 4096 in
theorem rootCandidates977_checked : candidates edge 5 0 7723 tail978 best.bound = rootCandidates977 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates976 : List Nat := [7963]
set_option maxRecDepth 4096 in
theorem rootCandidates976_checked : candidates edge 5 0 7717 tail977 best.bound = rootCandidates976 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
