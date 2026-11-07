import solvers.«98.py.lean.d».Correctness

namespace solvers.«98».pattern_signature
open FormalSW.PyAst HeapCalls

theorem «Returns» (text : String) :
    (HeapCalls.call program "pattern_signature" [.scalar (.string text)] []).Returns := by
  obtain ⟨_, _, returned, _⟩ := result text []
  exact returned.returns

theorem «Supported» (text : String) :
    (HeapCalls.call program "pattern_signature" [.scalar (.string text)] []).Supported :=
  («Returns» text).supported

/-- Two actual source calls yield equal tuple keys exactly when an injective substitution exists. -/
theorem pair_result (word digits : String) (heap : Mutable.Heap) :
    ∃ middle final left right,
      ReturnsValue program "pattern_signature" [.scalar (.string word)] heap (.ref left) middle ∧
      ReturnsValue program "pattern_signature" [.scalar (.string digits)] middle (.ref right) final ∧
      (final[left]? = final[right]? ↔
        ∃ f : Char → Char, word.toList.map f = digits.toList ∧
          Set.InjOn f {letter | letter ∈ word.toList}) := by
  obtain ⟨middle, left, first, firstFound, _⟩ := result word heap
  obtain ⟨final, right, second, secondFound, preserved⟩ := result digits middle
  obtain ⟨extra, rfl⟩ := preserved
  refine ⟨middle, middle ++ extra, left, right, first, second, ?_⟩
  have kept := ProjectEulerSolutions.HeapCollection.prefix_found middle extra left _ firstFound
  rw [kept, secondFound]
  have injective : Function.Injective number := by
    intro a b same
    have equal : (a : Int) = b := by simpa [number] using same
    omega
  simp only [Option.some.injEq, Mutable.Object.tuple.injEq]
  constructor
  · intro same
    have keys : ProjectEulerSolutions.AnagramSquares.labels [] word.toList =
        ProjectEulerSolutions.AnagramSquares.labels [] digits.toList :=
      (List.map_injective_iff.mpr injective) same
    exact (ProjectEulerSolutions.AnagramSquares.labels_eq_iff_substitution '0' _ _).mp keys
  · intro assignment
    have keys := (ProjectEulerSolutions.AnagramSquares.labels_eq_iff_substitution '0' _ _).mpr assignment
    simp [keys]

end solvers.«98».pattern_signature

namespace solvers.«98».squares_with_length
open FormalSW.PyAst HeapCalls
open pattern_signature (program number)

theorem «Returns» (width : Nat) (positive : 0 < width) :
    (HeapCalls.call program "squares_with_length" [number width] []).Returns := by
  obtain ⟨_, _, _, returned, _⟩ := result width positive []
  exact returned.returns

theorem «Supported» (width : Nat) (positive : 0 < width) :
    (HeapCalls.call program "squares_with_length" [number width] []).Supported :=
  («Returns» width positive).supported

end solvers.«98».squares_with_length

namespace solvers.«98».mapping
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.AnagramSquares
open data.«project-euler-statements».data.py.project_euler_statements.p98.mapping

/-- Actual optimized pattern calls and the complete reference dictionary checker accept the same candidates. -/
theorem checker_agrees (word digits : String) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (globals : Mutable.Globals) (otherwise : List Stmt)
    (sameLength : word.toList.length = digits.toList.length)
    (getWord : env "word" = some (.scalar (.string word))) (getSquare : env "square" = some (.scalar (.string digits))) :
    ∃ middle tested left right, ∃ accepted : Bool, ∃ finalEnv letters values garbage,
      ReturnsValue pattern_signature.program "pattern_signature" [.scalar (.string word)] heap (.ref left) middle ∧
      ReturnsValue pattern_signature.program "pattern_signature" [.scalar (.string digits)] middle (.ref right) tested ∧
      Pending env (prepareCode ++ [loopCode otherwise]) tested finalEnv (if accepted then otherwise else [])
        (work tested letters values garbage) globals ∧
      (accepted = true ↔ tested[left]? = tested[right]?) ∧
      (accepted = true → letters = alphabet [] word.toList ∧ values = alphabet [] digits.toList) := by
  obtain ⟨middle, tested, left, right, first, second, keys⟩ := pattern_signature.pair_result word digits heap
  obtain ⟨accepted, finalEnv, letters, values, garbage, checked, accepts, mapping, _⟩ :=
    result word digits tested env globals otherwise sameLength getWord getSquare
  refine ⟨middle, tested, left, right, accepted, finalEnv, letters, values, garbage, first, second, checked, ?_, mapping⟩
  exact accepts.trans ((labels_eq_iff_substitution '0' _ _).trans keys.symm)

end solvers.«98».mapping

namespace solvers.«98».solve
open FormalSW.PyAst HeapCalls
open pattern_signature (program)

theorem «Returns» (words : List String) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (words.map text))) :
    (HeapCalls.call program "solve" [.ref address] heap).Returns := by
  obtain ⟨_, _, returned, _⟩ := result words heap address found
  exact returned.returns

theorem «Supported» (words : List String) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (words.map text))) :
    (HeapCalls.call program "solve" [.ref address] heap).Supported :=
  («Returns» words heap address found).supported

/-- Both complete source calls agree for arbitrary word lists and preserve the caller's aliases and cycles. -/
theorem «Equiv(naive)» (words : List String) (heap : Mutable.Heap) (address : Nat)
    (closed : heap.Valid) (found : heap[address]? = some (.list (words.map text))) :
    (HeapCalls.call program "solve" [.ref address] heap).Equiv
      (HeapCalls.call P98Ast.module "naive" [.ref address] heap) := by
  obtain ⟨leftExtra, left⟩ := correct_result words heap address found
  obtain ⟨rightExtra, right⟩ := data.«project-euler-statements».data.py.project_euler_statements.p98.search.result words heap address found
  apply Mutable.Call.equiv_of_finishes left.finishes right.finishes
  exact Mutable.Completion.equivalent_allocations heap leftExtra rightExtra
    (mapped_word.number (ProjectEulerSolutions.AnagramSearch.solve words)) [] closed trivial (by intros; simp_all)

end solvers.«98».solve
