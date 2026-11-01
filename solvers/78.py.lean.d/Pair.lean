import solvers.«78.py.lean.d».Inner

/-! One actual source iteration accumulates both admissible pentagonal offsets. -/
namespace solvers.«78».inner
open FormalSW.PyAst HeapCalls
open ProjectEulerStatements.PartitionPentagonalLoop
set_option Elab.async false
variable {globals : Mutable.Globals}

theorem pair_result (boundName : String) (safe : boundName ∉ writes)
    (n k address : Nat) (values : Nat → Int) (items : List Mutable.Value)
    (heap : Heap) (env : HeapCalls.Env) (total : Int) (positive : 0 < k) (inside : first k ≤ n)
    (found : heap[address]? = some (.list items))
    (entries : ∀ j < n, items[j]? = some (number (values j)))
    (getBound : env boundName = some (number n)) (getK : env "k" = some (number k))
    (getFirst : env "g1" = some (number (first k)))
    (getP : env "p" = some (.ref address)) (getTotal : env "total" = some (number total)) :
    ∃ finalEnv, BlockContinues (globals := globals) program env ((body boundName).drop 2) heap finalEnv heap ∧
      finalEnv "k" = some (number (k+1 : Nat)) ∧
      finalEnv "total" = some (number (total + piece values n k)) ∧ Keeps env finalEnv := by
  let signedEnv := setLocal env "sign" (number (sign k))
  let firstTotal := total + sign k * values (n-first k)
  let firstEnv := setLocal signedEnv "total" (number firstTotal)
  let secondEnv := setLocal firstEnv "g2" (number (second k))
  let extraTotal := firstTotal + sign k * values (n-second k)
  let updatedEnv := if second k ≤ n then setLocal secondEnv "total" (number extraTotal) else secondEnv
  let finalEnv := setLocal updatedEnv "k" (number (k+1 : Nat))
  have signedKeeps : Keeps env signedEnv := Keeps.set env "sign" _ (by decide)
  have firstKeeps : Keeps signedEnv firstEnv := Keeps.set signedEnv "total" _ (by decide)
  have secondKeeps : Keeps firstEnv secondEnv := Keeps.set firstEnv "g2" _ (by decide)
  have kept : Keeps env secondEnv := (signedKeeps.trans firstKeeps).trans secondKeeps
  have firstRun : BlockContinues (globals := globals) program signedEnv
      [.augAssign (.name "total") .add (addExpression boundName "g1")] heap firstEnv heap :=
    add_result boundName "g1" n (first k) address values items heap signedEnv total (sign k)
      (first_positive k positive) inside found entries
      ((signedKeeps boundName safe).trans getBound)
      (by simpa [signedEnv, setLocal] using getFirst)
      (by simpa [signedEnv, setLocal] using getP) (by simp [signedEnv, setLocal])
      (by simpa [signedEnv, setLocal] using getTotal)
  have getSecondK : secondEnv "k" = some (number k) := by
    simpa [secondEnv, firstEnv, signedEnv, setLocal] using getK
  have secondRun : BlockContinues (globals := globals) program firstEnv
      [.assign [.name "g2"] (offsetExpression true)] heap secondEnv heap :=
    BlockContinues.assign (offset_result true k heap firstEnv
      (by simpa [firstEnv, signedEnv, setLocal] using getK))
  have secondTest : Evaluates (globals := globals) program secondEnv
      (.compare (.name "g2") [(.ltE, .name boundName)]) heap
      (flag (decide (second k ≤ n))) heap := by
    apply Evaluates.compare (av := number (second k)) (bv := number n)
      (middle := heap) (afterRight := heap) (finalHeap := heap)
      (Evaluates.lookup (by simp [secondEnv, setLocal]))
      (Evaluates.lookup ((kept boundName safe).trans getBound))
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  have optional : BlockContinues (globals := globals) program secondEnv
      [.ifThen (.compare (.name "g2") [(.ltE, .name boundName)])
        [.augAssign (.name "total") .add (addExpression boundName "g2")] []] heap updatedEnv heap := by
    by_cases extra : second k ≤ n
    · simp only [updatedEnv, if_pos extra]
      apply BlockContinues.branch (flag := true)
        (by simpa only [extra, decide_true] using secondTest) rfl
      exact add_result boundName "g2" n (second k) address values items heap secondEnv firstTotal (sign k)
        (by have := second_gt k positive; have := first_positive k positive; omega) extra found entries
        ((kept boundName safe).trans getBound) (by simp [secondEnv, setLocal])
        ((kept "p" (by decide)).trans getP) (by simp [secondEnv, firstEnv, signedEnv, setLocal])
        (by simp [secondEnv, firstEnv, setLocal])
    · simp only [updatedEnv, if_neg extra]
      exact BlockContinues.branch (flag := false)
        (by simpa only [extra, decide_false] using secondTest) rfl .nil
  have optionalKeeps : Keeps secondEnv updatedEnv := by
    dsimp [updatedEnv]
    split_ifs
    · exact Keeps.set secondEnv "total" _ (by decide)
    · exact Keeps.refl secondEnv
  have getUpdatedK : updatedEnv "k" = some (number k) := by
    dsimp [updatedEnv]
    split_ifs <;> simpa [setLocal] using getSecondK
  have increment : BlockContinues (globals := globals) program updatedEnv
      [.augAssign (.name "k") .add (.constant (.int 1))] heap finalEnv heap := by
    apply BlockContinues.augAssign getUpdatedK Evaluates.constant
    simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
  refine ⟨finalEnv, ?_, by simp [finalEnv, setLocal], ?_,
    (kept.trans optionalKeeps).trans (Keeps.set updatedEnv "k" _ (by decide))⟩
  · exact (BlockContinues.assign (name := "sign") (sign_result k heap env getK)).append
      (firstRun.append (secondRun.append (optional.append increment)))
  · change updatedEnv "total" = some (number (total + piece values n k))
    dsimp [updatedEnv]
    split_ifs with extra
    · simp [setLocal, extraTotal, firstTotal, piece, inside, extra, add_assoc]
    · simp [secondEnv, firstEnv, setLocal, firstTotal, piece, inside, extra]

end solvers.«78».inner
