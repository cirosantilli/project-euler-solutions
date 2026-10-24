import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates935 : List Nat := [7489, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates935_checked : candidates edge 5 0 7369 tail936 best.bound = rootCandidates935 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates934 : List Nat := [7699, 7723, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates934_checked : candidates edge 5 0 7351 tail935 best.bound = rootCandidates934 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates933 : List Nat := [8147]
set_option maxRecDepth 4096 in
theorem rootCandidates933_checked : candidates edge 5 0 7349 tail934 best.bound = rootCandidates933 := by
  unfold candidates
  have row : edge 7349 = SearchData.row7349.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates932 : List Nat := [7393, 7753, 8059]
set_option maxRecDepth 4096 in
theorem rootCandidates932_checked : candidates edge 5 0 7333 tail933 best.bound = rootCandidates932 := by
  unfold candidates
  have row : edge 7333 = SearchData.row7333.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates931 : List Nat := [7487, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates931_checked : candidates edge 5 0 7331 tail932 best.bound = rootCandidates931 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates930 : List Nat := [7369, 8179, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates930_checked : candidates edge 5 0 7321 tail931 best.bound = rootCandidates930 := by
  unfold candidates
  have row : edge 7321 = SearchData.row7321.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates929 : List Nat := [7963, 7993, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates929_checked : candidates edge 5 0 7309 tail930 best.bound = rootCandidates929 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates928 : List Nat := [7547, 8387]
set_option maxRecDepth 4096 in
theorem rootCandidates928_checked : candidates edge 5 0 7307 tail929 best.bound = rootCandidates928 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
