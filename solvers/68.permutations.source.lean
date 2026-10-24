import solvers.«68.candidate.source»
import lang.py.FormalSW.PermutationCalls

namespace solvers.«68»
open FormalSW.PyAst HeapCalls ProjectEulerStatements MagicRing MagicRingTextCalls
set_option Elab.async false

def capturedInnerRows (rows : List (List Nat)) : List (List SequenceSource) :=
  rows.map (fun row => (encode row).map sequenceSource)

def PermutationKeeps (before after : HeapCalls.Env) : Prop :=
  ∀ name, name ≠ "inner" → name ≠ "outer" → name ≠ "index" → name ≠ "value" →
    name ≠ "start" → name ≠ "text" → name ≠ "best" → after name = before name

abbrev innerLoop : Stmt := .forIn (.name "inner")
  (.permutationsIterable [.name "inner_set", .constant .none]) innerPermutationBody []

theorem inner_loop_source : combinationBody[5]? = some innerLoop := rfl

theorem permutations_result (n : Nat) (positive : 0 < n) (digits : Int)
    (selected : List Nat) (size : selected.length = n) (distinct : selected.Nodup)
    (subset : ∀ value ∈ selected, value ∈ labels n) (rows : List (List Nat))
    (valid : ∀ row ∈ rows, row.Perm selected) (best : String)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (remainingAddress : Nat)
    (getRemaining : heap[remainingAddress]? = some (.set
      (ScalarSet.difference (IntegerSets.entries (ints (labels n)))
        (IntegerSets.entries (ints selected)))))
    (localRemaining : env "remaining_outer" = some (.ref remainingAddress))
    (getN : env "n" = some (number n))
    (getTarget : env "target" = some (number (target n selected)))
    (getDigits : env "digits" = some (.scalar (.int digits)))
    (getBest : env "best" = some (string best))
    (following : List Stmt) (answer : Mutable.Value)
    (post : ∀ nextEnv extra,
      nextEnv "best" = some (string (innerScan n digits rows best)) →
      PermutationKeeps env nextEnv →
      ∃ finalHeap, BlockReturns program nextEnv following (heap ++ extra) answer finalHeap) :
    ∃ finalHeap, BlockReturns program env
      (.permutationStepLoop "inner" (capturedInnerRows rows) innerPermutationBody [] :: following)
      heap answer finalHeap := by
  induction rows generalizing best heap env with
  | nil =>
      obtain ⟨finalHeap, continued⟩ := post env [] getBest (fun _ _ _ _ _ _ _ _ => rfl)
      exact ⟨finalHeap, BlockReturns.permutation_done
        (by simpa only [List.append_nil] using continued)⟩
  | cons row rest ih =>
      have rowPerm := valid row List.mem_cons_self
      have rowSize : row.length = n := rowPerm.length_eq.trans size
      have nonempty : (encode row).isEmpty = false := by
        apply List.isEmpty_eq_false_iff.mpr
        intro empty
        have length := congrArg List.length empty
        simp only [encode, List.length_map, List.length_nil] at length
        omega
      let rowHeap := heap ++ [.tuple (encode row)]
      let rowEnv := setLocal env "inner" (.ref heap.length)
      let next : Stmt :=
        .permutationStepLoop "inner" (capturedInnerRows rest) innerPermutationBody []
      have remainingInTuple : rowHeap[remainingAddress]? = some (.set
          (ScalarSet.difference (IntegerSets.entries (ints (labels n)))
            (IntegerSets.entries (ints selected)))) := by
        dsimp [rowHeap]
        rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp getRemaining).1]
        exact getRemaining
      have postRow (nextEnv : HeapCalls.Env) (extra : Mutable.Heap)
          (finalBest : nextEnv "best" = some (string (innerVisit n digits best row)))
          (candidateKept : CandidateKeeps rowEnv nextEnv) :
          ∃ finalHeap, BlockReturns program nextEnv (next :: following)
            (rowHeap ++ extra) answer finalHeap := by
        have stepKept : PermutationKeeps env nextEnv := by
          intro name notInner notOuter notIndex notValue notStart notText notBest
          rw [candidateKept name notOuter notIndex notValue notStart notText notBest]
          simp [rowEnv, setLocal, notInner]
        have remainingNext : (rowHeap ++ extra)[remainingAddress]? = some (.set
            (ScalarSet.difference (IntegerSets.entries (ints (labels n)))
              (IntegerSets.entries (ints selected)))) := by
          rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp remainingInTuple).1]
          exact remainingInTuple
        obtain ⟨finalHeap, continued⟩ := ih
          (valid := fun entry member => valid entry (List.mem_cons_of_mem _ member))
          (best := innerVisit n digits best row) (heap := rowHeap ++ extra) (env := nextEnv)
          (getRemaining := remainingNext)
          (localRemaining := by
            rw [stepKept "remaining_outer" (by decide) (by decide) (by decide) (by decide)
              (by decide) (by decide) (by decide)]
            exact localRemaining)
          (getN := by
            rw [stepKept "n" (by decide) (by decide) (by decide) (by decide)
              (by decide) (by decide) (by decide)]
            exact getN)
          (getTarget := by
            rw [stepKept "target" (by decide) (by decide) (by decide) (by decide)
              (by decide) (by decide) (by decide)]
            exact getTarget)
          (getDigits := by
            rw [stepKept "digits" (by decide) (by decide) (by decide) (by decide)
              (by decide) (by decide) (by decide)]
            exact getDigits)
          (getBest := finalBest)
          (post := by
            intro finalEnv more nextBest tailKept
            have allKept : PermutationKeeps env finalEnv := by
              intro name notInner notOuter notIndex notValue notStart notText notBest
              rw [tailKept name notInner notOuter notIndex notValue notStart notText notBest,
                stepKept name notInner notOuter notIndex notValue notStart notText notBest]
            obtain ⟨finalHeap, finished⟩ := post finalEnv (.tuple (encode row) :: (extra ++ more))
              (by simpa only [innerScan_cons] using nextBest) allKept
            exact ⟨finalHeap, by simpa only [rowHeap, List.append_assoc, List.singleton_append]
              using finished⟩)
        exact ⟨finalHeap, continued⟩
      obtain ⟨finalHeap, advanced⟩ := candidate_result n positive digits best row rowSize
        (rowPerm.symm.nodup distinct)
        (fun value member => subset value (rowPerm.mem_iff.mp member)) rowHeap rowEnv
        heap.length remainingAddress (by simp [rowHeap])
        (by simpa only [complement_perm n selected row rowPerm.symm] using remainingInTuple)
        rfl (by simpa [rowEnv, setLocal] using localRemaining)
        (by simpa [rowEnv, setLocal] using getN)
        (by simpa [rowEnv, setLocal, target, rowPerm.sum_eq] using getTarget)
        (by simpa [rowEnv, setLocal] using getDigits)
        (by simpa [rowEnv, setLocal] using getBest) next following answer postRow
      refine ⟨finalHeap, ?_⟩
      apply BlockReturns.permutation_step (tuple := .ref heap.length) (middle := rowHeap)
      · simp [List.map_map, Function.comp_def, Mutable.allocateTuple, nonempty, rowHeap,
          Except.map]
      · exact advanced

theorem inner_loop_result (n : Nat) (positive : 0 < n) (digits : Int)
    (selected : List Nat) (size : selected.length = n) (distinct : selected.Nodup)
    (subset : ∀ value ∈ selected, value ∈ labels n) (best : String)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (selectedAddress remainingAddress : Nat)
    (getSelected : heap[selectedAddress]? = some (.tuple (encode selected)))
    (getRemaining : heap[remainingAddress]? = some (.set
      (ScalarSet.difference (IntegerSets.entries (ints (labels n)))
        (IntegerSets.entries (ints selected)))))
    (localSelected : env "inner_set" = some (.ref selectedAddress))
    (localRemaining : env "remaining_outer" = some (.ref remainingAddress))
    (getN : env "n" = some (number n))
    (getTarget : env "target" = some (number (target n selected)))
    (getDigits : env "digits" = some (.scalar (.int digits)))
    (getBest : env "best" = some (string best))
    (following : List Stmt) (answer : Mutable.Value)
    (post : ∀ nextEnv extra,
      nextEnv "best" = some (string (innerScan n digits (Permutations.rows n selected) best)) →
      PermutationKeeps env nextEnv →
      ∃ finalHeap, BlockReturns program nextEnv following (heap ++ extra) answer finalHeap) :
    ∃ finalHeap, BlockReturns program env (innerLoop :: following) heap answer finalHeap := by
  obtain ⟨finalHeap, scanned⟩ := permutations_result n positive digits selected size distinct subset
    (Permutations.rows n selected)
    (fun row member => Permutations.row_perm n selected row size member)
    best heap env remainingAddress getRemaining localRemaining getN getTarget getDigits getBest
    following answer post
  refine ⟨finalHeap, ?_⟩
  apply BlockReturns.for_permutations
    (.cons (Evaluates.lookup localSelected) (.cons Evaluates.constant .nil)) _ scanned
  simp [permutationRows, Permutations.acquire, Mutable.sequence, getSelected, Permutations.size,
    encode, size, Permutations.rows_map, capturedInnerRows, List.map_map, Function.comp_def,
    Except.map]

end solvers.«68»
