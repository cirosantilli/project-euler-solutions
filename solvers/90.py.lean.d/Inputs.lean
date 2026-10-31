import solvers.«90.py.lean.d».Equivalence

namespace solvers.«90».source_equivalence
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

def standardPairs (pairs : List (Nat × Nat)) : List ReferencePair :=
  pairs.zipIdx.map (fun entry => ⟨entry.1.1, entry.1.2, entry.2⟩)

def standardHeap (pairs : List (Nat × Nat)) : Mutable.Heap :=
  pairs.map (fun pair => .tuple [number pair.1, number pair.2]) ++
    [.list ((standardPairs pairs).map (fun pair => .ref pair.address))]

def standardRequired (pairs : List (Nat × Nat)) : Required := ⟨standardPairs pairs, pairs.length⟩

theorem standard_model (pairs : List (Nat × Nat)) :
    (standardRequired pairs).pairs.map
      data.«project-euler-statements».data.py.project_euler_statements.p90.can_show.Pair.model = pairs := by
  simpa only [standardRequired, standardPairs, List.map_map, Function.comp_def,
    data.«project-euler-statements».data.py.project_euler_statements.p90.can_show.Pair.model] using
    List.zipIdx_map_fst 0 pairs

/-- Every finite requirement list admits a valid entry state, including the empty list and repeated pairs. -/
theorem standard_domain (pairs : List (Nat × Nat)) :
    Domain (standardHeap pairs) [] (standardRequired pairs) := by
  constructor
  · intro object member
    simp only [standardHeap, List.mem_append, List.mem_singleton] at member
    rcases member with tuple | rfl
    · obtain ⟨pair, _, rfl⟩ := List.mem_map.mp tuple
      simp [number]
    · intro value member
      obtain ⟨pair, pairMember, rfl⟩ := List.mem_map.mp member
      obtain ⟨entry, inZip, rfl⟩ := List.mem_map.mp pairMember
      have bound := List.snd_lt_of_mem_zipIdx inZip
      change entry.2 < (standardHeap pairs).length
      simp only [standardHeap, List.length_append, List.length_map, List.length_singleton]
      omega
  · simp
  · constructor
    · simp [standardRequired, standardHeap,
        data.«project-euler-statements».data.py.project_euler_statements.p90.can_show.Pair.reference]
    · intro pair member
      obtain ⟨entry, inZip, rfl⟩ := List.mem_map.mp member
      have found := List.mem_zipIdx_iff_getElem?.mp inZip
      have tupleFound : (pairs.map (fun pair => Mutable.Object.tuple [number pair.1, number pair.2]))[entry.2]? =
          some (.tuple [number entry.1.1, number entry.1.2]) := by simp [found]
      exact HeapCollection.prefix_found _ _ entry.2 _ tupleFound

end solvers.«90».source_equivalence
