import solvers.«96.py.lean.d».CandidateSolved
import ProjectEulerSolutions.Lib.SudokuCandidateChoice
import lang.py.FormalSW.TupleComparisons

namespace solvers.«96».solve_cands.choice
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

abbrev count : Expr := .builtinCall .intBitCount [.subscript (.name "cands") (.name "i")]
abbrev filter : Expr := .compare count [(.gt, .constant (.int 1))]
abbrev element : Expr := .tuple [count, .name "i"]
abbrev generators : List (Expr × Expr × List Expr) :=
  [(.name "i", .call (.name "range") [.constant (.int 81)], [filter])]
abbrev expression : Expr := .consumeGenerator (extremumConsumer false none) element generators
abbrev statement : Stmt := .assign [.tuple [.name "_", .name "idx"]] expression
abbrev body : List Stmt := comprehensionFilters [filter] (extremumBody false element)
abbrev pair (candidates : Candidates) (slot : Fin 81) : List Mutable.Value :=
  [number (Choice.count candidates slot), number slot.val]

theorem source : solved.definition.2[2]? = some statement := rfl

theorem counted (address : Nat) (candidates : Candidates) (slot : Fin 81) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (found : View address heap candidates)
    (getCandidates : env "cands" = some (.ref address)) (getIndex : env "i" = some (number slot.val)) :
    Evaluates program env count heap (number (Choice.count candidates slot)) heap globals := by
  have read := Evaluates.subscript (globals := globals) (module := program)
    (by intros; simp) (Evaluates.lookup getCandidates) (Evaluates.lookup getIndex)
    (get_mask address heap candidates slot found)
  apply Evaluates.builtin (.cons read .nil)
  change (some (.ok (number (FormalSW.Py.Builtin.Int.bitCount
      (Masks.encode (candidates (Encoding.cell slot))))), heap) :
    Option (Except ExceptionObject Mutable.Value × Mutable.Heap)) = _
  simp only [Masks.bitCount, Choice.count]

theorem filtered (address : Nat) (candidates : Candidates) (slot : Fin 81) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (found : View address heap candidates)
    (getCandidates : env "cands" = some (.ref address)) (getIndex : env "i" = some (number slot.val)) :
    Evaluates program env filter heap (.scalar (.bool (decide (1 < Choice.count candidates slot)))) heap globals := by
  apply Evaluates.compare (counted address candidates slot env heap globals found getCandidates getIndex)
    Evaluates.constant
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  omega

theorem yielded (address : Nat) (candidates : Candidates) (slot : Fin 81) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (found : View address heap candidates)
    (getCandidates : env "cands" = some (.ref address)) (getIndex : env "i" = some (number slot.val)) :
    Evaluates program env element heap (.ref heap.length) (heap ++ [.tuple (pair candidates slot)]) globals :=
  Evaluates.tuple (.cons (counted address candidates slot env heap globals found getCandidates getIndex)
    (.cons (Evaluates.lookup getIndex) .nil)) (by simp)

theorem compared (candidates : Candidates) (left right : Fin 81) (heap : Mutable.Heap) (a b : Nat)
    (leftFound : heap[a]? = some (.tuple (pair candidates left)))
    (rightFound : heap[b]? = some (.tuple (pair candidates right))) :
    HeapCalls.compare .lt (.ref a) (.ref b) heap =
      some (.ok (.scalar (.bool (decide (Choice.priority candidates left < Choice.priority candidates right)))), heap) := by
  have lexicographic : HeapSort.lexInt [(Choice.count candidates left : Int), left.val]
      [(Choice.count candidates right : Int), right.val] =
      decide (Choice.priority candidates left < Choice.priority candidates right) := by
    apply Bool.eq_iff_iff.mpr
    simp only [HeapSort.lexInt, Bool.false_eq_true, decide_eq_true_eq, ite_eq_iff,
      Choice.priority_lt]
    simp only [Int.ofNat_inj, Int.ofNat_lt]
    constructor
    · rintro (⟨same, impossible | ⟨_, smaller⟩⟩ | ⟨_, smaller⟩)
      · exact False.elim impossible.2
      · exact Or.inr ⟨same, smaller⟩
      · exact Or.inl smaller
    · rintro (smaller | ⟨same, smaller⟩)
      · exact Or.inr ⟨Nat.ne_of_lt smaller, smaller⟩
      · exact Or.inl ⟨same, Or.inr ⟨Nat.ne_of_lt smaller, smaller⟩⟩
  simpa only [lexicographic] using compare_int_tuple_lt heap a b
    [(Choice.count candidates left : Int), left.val] [(Choice.count candidates right : Int), right.val]
    leftFound rightFound

def State (candidates : Candidates) (heap : Mutable.Heap) (env : HeapCalls.Env) (best : Option (Fin 81)) : Prop :=
  env extremumSeen = some (.scalar (.bool best.isSome)) ∧ match best with
    | none => True
    | some slot => ∃ address, env extremumBest = some (.ref address) ∧ heap[address]? = some (.tuple (pair candidates slot))

theorem State.append {candidates : Candidates} {heap : Mutable.Heap} {env : HeapCalls.Env} {best : Option (Fin 81)}
    (known : State candidates heap env best) (extra : Mutable.Heap) : State candidates (heap ++ extra) env best := by
  refine ⟨known.1, ?_⟩
  cases best with
  | none => trivial
  | some slot =>
      obtain ⟨address, lookup, found⟩ := known.2
      exact ⟨address, lookup, by rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1, found]⟩

theorem State.bind_index {candidates : Candidates} {heap : Mutable.Heap} {env : HeapCalls.Env} {best : Option (Fin 81)}
    (known : State candidates heap env best) (index : Nat) :
    State candidates heap (setLocal env "i" (number index)) best := by
  refine ⟨by simpa [setLocal, extremumSeen] using known.1, ?_⟩
  cases best with
  | none => trivial
  | some slot =>
      obtain ⟨address, lookup, found⟩ := known.2
      exact ⟨address, by simpa [setLocal, extremumBest] using lookup, found⟩

/-- Each yielded pair is compared in Python's tuple order and keeps its original heap identity. -/
theorem eligible (address : Nat) (candidates : Candidates) (slot : Fin 81) (best : Option (Fin 81))
    (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals) (found : View address heap candidates)
    (getCandidates : env "cands" = some (.ref address)) (getIndex : env "i" = some (number slot.val))
    (known : State candidates heap env best) (multiple : 1 < Choice.count candidates slot) :
    ∃ updated, BlockContinues program env (extremumBody false element) heap updated
        (heap ++ [.tuple (pair candidates slot)]) globals ∧
      State candidates (heap ++ [.tuple (pair candidates slot)]) updated (Choice.consider candidates best slot) ∧
      updated "cands" = env "cands" := by
  let after := heap ++ [.tuple (pair candidates slot)]
  let candidateEnv := setLocal env extremumCandidate (.ref heap.length)
  have assigned : BlockContinues program env [.assign [.name extremumCandidate] element] heap candidateEnv after globals :=
    BlockContinues.assign (yielded address candidates slot env heap globals found getCandidates getIndex)
  have candidateFound : candidateEnv extremumCandidate = some (.ref heap.length) := by simp [candidateEnv, setLocal]
  have seenFound : candidateEnv extremumSeen = some (.scalar (.bool best.isSome)) := by
    simpa [candidateEnv, setLocal, extremumSeen, extremumCandidate] using known.1
  have tupleFound : after[heap.length]? = some (.tuple (pair candidates slot)) := by simp [after]
  cases best with
  | none =>
      let updated := setLocal (setLocal candidateEnv extremumBest (.ref heap.length)) extremumSeen (.scalar (.bool true))
      refine ⟨updated, assigned.append ?_, ?_, ?_⟩
      · apply BlockContinues.branch (flag := false) (Evaluates.lookup seenFound) rfl
        exact (BlockContinues.assign (Evaluates.lookup candidateFound)).append (BlockContinues.assign Evaluates.constant)
      · simp only [Choice.consider, if_pos multiple, State, Option.isSome_some]
        refine ⟨by simp [updated, setLocal], heap.length, ?_, tupleFound⟩
        simp [updated, setLocal, extremumBest, extremumSeen]
      · simp [updated, candidateEnv, setLocal, extremumBest, extremumSeen, extremumCandidate]
  | some old =>
      obtain ⟨oldAddress, oldLookup, oldFound⟩ := known.2
      have bestFound : candidateEnv extremumBest = some (.ref oldAddress) := by
        simpa [candidateEnv, setLocal, extremumBest, extremumCandidate] using oldLookup
      have oldTuple : after[oldAddress]? = some (.tuple (pair candidates old)) := by
        rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp oldFound).1, oldFound]
      have comparison : Evaluates program candidateEnv
          (extremumComparison (.compare (.name extremumCandidate) [(.lt, .name extremumBest)]))
          after (.scalar (.bool (decide (Choice.priority candidates slot < Choice.priority candidates old)))) after globals :=
        Evaluates.extremum_comparison (Evaluates.compare (Evaluates.lookup candidateFound) (Evaluates.lookup bestFound)
          (compared candidates slot old after heap.length oldAddress tupleFound oldTuple))
      by_cases better : Choice.priority candidates slot < Choice.priority candidates old
      · let updated := setLocal candidateEnv extremumBest (.ref heap.length)
        refine ⟨updated, assigned.append ?_, ?_, ?_⟩
        · apply BlockContinues.branch (flag := true) (Evaluates.lookup seenFound) rfl
          apply BlockContinues.branch (flag := true) comparison (by simp [better, Mutable.truth, scalarTruth])
          exact BlockContinues.assign (Evaluates.lookup candidateFound)
        · simp only [Choice.consider, if_pos multiple, if_pos better, State, Option.isSome_some]
          refine ⟨?_, heap.length, by simp [updated, setLocal], tupleFound⟩
          simpa [updated, setLocal, extremumSeen, extremumBest] using seenFound
        · simp [updated, candidateEnv, setLocal, extremumBest, extremumCandidate]
      · refine ⟨candidateEnv, assigned.append ?_, ?_, ?_⟩
        · apply BlockContinues.branch (flag := true) (Evaluates.lookup seenFound) rfl
          exact BlockContinues.branch (flag := false) comparison (by simp [better, Mutable.truth, scalarTruth]) .nil
        · simp only [Choice.consider, if_pos multiple, if_neg better, State, Option.isSome_some]
          exact ⟨seenFound, oldAddress, bestFound, oldTuple⟩
        · simp [candidateEnv, setLocal, extremumCandidate]

end
end solvers.«96».solve_cands.choice
