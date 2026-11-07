import solvers.«98.py.lean.d».Substitution

namespace solvers.«98».mapped_word
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open pattern_signature (program)
open squares_with_length (squares setValues)
set_option Elab.async false

abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
def integerValue (text : String) : Nat := Nat.ofDigitChars 10 text.toList 0
abbrev squareTest : Expr := .membership (.name "num2") (.attribute (.name "sqinfo") "as_set") false
abbrev improve (name : String) : Stmt :=
  .ifThen (.compare (.name name) [(.gt, .name "best")]) [.assign [.name "best"] (.name name)] []
abbrev update : List Stmt :=
  [.assign [.name "sq2"] (.builtinCall .int [.name "num2"]), improve "sq1", improve "sq2"]

theorem source_update : body.drop 2 = [.ifThen squareTest update []] := rfl

theorem member_result (width : Int) (outer address : Nat) (text : String) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (stored : square_cache.Stored outer heap width address)
    (getRecord : env "sqinfo" = some (.ref address)) (getText : env "num2" = some (.scalar (.string text))) :
    Evaluates program env squareTest heap (.scalar (.bool (decide (text ∈ squares width.toNat)))) heap := by
  obtain ⟨dictionary, _, _, getSet, _, _, found⟩ := square_cache.fields heap outer width address env stored getRecord
  apply Evaluates.membership (.cons (Evaluates.lookup getText) (.cons getSet .nil))
  simp [HeapCalls.membership, found, squares_with_length.contains_setValues]

theorem integer_result (width : Nat) (name text : String) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (member : text ∈ squares width) (getText : env name = some (.scalar (.string text))) :
    Evaluates program env (.builtinCall .int [.name name]) heap (number (integerValue text)) heap := by
  obtain ⟨root, _, _, rfl⟩ := (squares_with_length.mem_squares width text).mp member
  have parsed : FormalSW.Py.Builtin.Int.parseText (root * root).repr = .ok (root * root : Int) :=
    FormalSW.Py.Builtin.Int.parseText_nat (root * root)
  apply Evaluates.builtin (.cons (Evaluates.lookup getText) .nil)
  simp [primitive, Calls.integerConversion, parsed,
    integerValue, AnagramSearch.squareText, Nat.toList_repr]
  rfl

theorem improve_result (name : String) (n best : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getNumber : env name = some (number n)) (getBest : env "best" = some (number best)) :
    ∃ finalEnv, BlockContinues program env [improve name] heap finalEnv heap ∧
      finalEnv "best" = some (number (max best n)) ∧
      (∀ other, other ≠ "best" → finalEnv other = env other) := by
  have tested : Evaluates program env (.compare (.name name) [(.gt, .name "best")]) heap
      (.scalar (.bool (decide (best < n)))) heap := by
    apply Evaluates.compare (Evaluates.lookup getNumber) (Evaluates.lookup getBest)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases larger : best < n
  · refine ⟨setLocal env "best" (number n), ?_, ?_, ?_⟩
    · apply BlockContinues.branch (flag := true) tested (by simp [Mutable.truth, scalarTruth, larger])
      exact BlockContinues.assign (Evaluates.lookup getNumber)
    · simp [setLocal, max_eq_right (le_of_lt larger)]
    · intro other different; simp [setLocal, different]
  · refine ⟨env, ?_, ?_, by intros; rfl⟩
    · exact BlockContinues.branch (flag := false) tested (by simp [Mutable.truth, scalarTruth, larger]) BlockContinues.nil
    · simpa [max_eq_left (le_of_not_gt larger)] using getBest

/-- Actual set membership controls parsing and both source maximum updates. -/
theorem update_result (width : Int) (outer address : Nat) (text : String) (firstSquare best : Int)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (stored : square_cache.Stored outer heap width address)
    (getRecord : env "sqinfo" = some (.ref address)) (getText : env "num2" = some (.scalar (.string text)))
    (getFirst : env "sq1" = some (number firstSquare)) (getBest : env "best" = some (number best)) :
    ∃ finalEnv, BlockContinues program env (body.drop 2) heap finalEnv heap ∧
      finalEnv "best" = some (number (if text ∈ squares width.toNat then max (max best firstSquare) (integerValue text) else best)) ∧
      (∀ name, name ∉ ["sq2", "best"] → finalEnv name = env name) := by
  by_cases member : text ∈ squares width.toNat
  · let ready := setLocal env "sq2" (number (integerValue text))
    obtain ⟨firstEnv, first, firstBest, firstKept⟩ := improve_result "sq1" firstSquare best ready heap
      (by simpa [ready, setLocal] using getFirst) (by simpa [ready, setLocal] using getBest)
    obtain ⟨finalEnv, second, secondBest, secondKept⟩ := improve_result "sq2" (integerValue text) (max best firstSquare) firstEnv heap
      ((firstKept "sq2" (by decide)).trans (by simp [ready, setLocal])) firstBest
    refine ⟨finalEnv, ?_, ?_, ?_⟩
    · rw [source_update]
      apply BlockContinues.branch (flag := true) (member_result width outer address text heap env stored getRecord getText)
        (by simp [Mutable.truth, scalarTruth, member])
      exact (BlockContinues.assign (integer_result width.toNat "num2" text heap env member getText)).append (first.append second)
    · simpa only [if_pos member] using secondBest
    · intro name absent
      have notBest : name ≠ "best" := fun same => absent (by simp [same])
      have notSquare : name ≠ "sq2" := fun same => absent (by simp [same])
      rw [secondKept name notBest, firstKept name notBest]
      simp [ready, setLocal, notSquare]
  · refine ⟨env, ?_, ?_, by intros; rfl⟩
    · rw [source_update]
      exact BlockContinues.branch (flag := false) (member_result width outer address text heap env stored getRecord getText)
        (by simp [Mutable.truth, scalarTruth, member]) BlockContinues.nil
    · simpa only [if_neg member] using getBest

end solvers.«98».mapped_word
