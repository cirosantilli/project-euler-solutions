import solvers.«78.loop.source»
import ProjectEulerStatements.Lib.PartitionPentagonalLoop

/-! A source iteration appends exactly the next partition residue to its private table. -/
namespace solvers.«78».solve
open FormalSW.PyAst HeapCalls
open ProjectEulerStatements.PartitionPentagonalLoop ProjectEulerStatements.PartitionRecurrence
set_option Elab.async false

def encode (values : List Int) : List Mutable.Value := values.map number
def entryValue (values : List Int) (j : Nat) : Int := values[j]?.getD 0
def working (base : Heap) (values : List Int) : Heap := base ++ [.list (encode values)]

@[simp] theorem found (base : Heap) (values : List Int) :
    (working base values)[base.length]? = some (.list (encode values)) := by simp [working]

theorem table_entry (values : List Int) (j : Nat) (inside : j < values.length) :
    (encode values)[j]? = some (number (entryValue values j)) := by
  simp [encode, entryValue, inside]

def Valid (modulus : Int) (n : Nat) (values : List Int) : Prop :=
  values.length = n ∧ ∀ j < n, entryValue values j % modulus = count j % modulus

theorem valid_initial (modulus : Int) : Valid modulus 1 [1] := by
  refine ⟨rfl, ?_⟩
  intro j inside
  have zero : j = 0 := by omega
  subst j
  simp [entryValue, count]

theorem valid_append (modulus : Int) (n : Nat) (values : List Int) (valid : Valid modulus n values) :
    Valid modulus (n+1) (values ++ [count n % modulus]) := by
  have length := valid.1
  refine ⟨by simp [valid.1], ?_⟩
  intro j inside
  by_cases earlier : j < n
  · have unchanged : entryValue (values ++ [count n % modulus]) j = entryValue values j := by
      simp only [entryValue, List.getElem?_append_left (by omega : j < values.length)]
    rw [unchanged]
    exact valid.2 j earlier
  · have same : j = n := by omega
    subst j
    simp [entryValue, valid.1]

theorem iteration_result (modulus : Int) (positive : 0 < modulus) (n : Nat) (lower : 0 < n)
    (values : List Int) (valid : Valid modulus n values) (base : Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) (getMod : env "mod" = some (number modulus))
    (getP : env "p" = some (.ref base.length)) :
    ∃ finalEnv, BlockContinues program env (inner.solveBody.take 5) (working base values) finalEnv
        (working base (values ++ [count n % modulus])) ∧
      finalEnv "total" = some (number (count n % modulus)) ∧ inner.Keeps env finalEnv := by
  have length := valid.1
  let totalEnv := setLocal env "total" (number 0)
  let counterEnv := setLocal totalEnv "k" (number 1)
  have totalKeeps : inner.Keeps env totalEnv := inner.Keeps.set env "total" _ (by decide)
  have counterKeeps : inner.Keeps totalEnv counterEnv := inner.Keeps.set totalEnv "k" _ (by decide)
  have initialized := totalKeeps.trans counterKeeps
  obtain ⟨middleEnv, scanned, totalValue, scanKeeps⟩ :=
    inner.loop_result "n" (by decide) n 1 base.length (entryValue values) (encode values)
      (working base values) counterEnv 0 (by omega) (found base values)
      (fun j inside => table_entry values j (by omega))
      ((initialized "n" (by decide)).trans getN) (by simp [counterEnv, setLocal])
      ((initialized "p" (by decide)).trans getP) (by simp [counterEnv, totalEnv, setLocal])
  simp only [zero_add] at totalValue
  have kept := initialized.trans scanKeeps
  have residue := tail_start_mod (entryValue values) n modulus valid.2 lower
  let reduced := setLocal middleEnv "total" (number (count n % modulus))
  have reducedKeeps : inner.Keeps middleEnv reduced := inner.Keeps.set middleEnv "total" _ (by decide)
  have allKept := kept.trans reducedKeeps
  have reducedTotal : reduced "total" = some (number (count n % modulus)) := by simp [reduced, setLocal]
  have reducedRun : BlockContinues program middleEnv
      [.augAssign (.name "total") .mod (.name "mod")] (working base values) reduced (working base values) := by
    apply BlockContinues.augAssign totalValue (Evaluates.lookup ((kept "mod" (by decide)).trans getMod))
    simp [HeapCalls.inplace, Mutable.binary, scalarBinary, positive, show modulus ≠ 0 by omega, residue]
  have appended : BlockContinues program reduced
      [.expr (.call (.attribute (.name "p") "append") [.name "total"])]
      (working base values) reduced (working base (values ++ [count n % modulus])) := by
    have stored : (working base values).set base.length
        (.list (encode values ++ [number (count n % modulus)])) =
          working base (values ++ [count n % modulus]) := by
      simp [working, encode, List.map_append, List.set_append_right]
    rw [← stored]
    exact BlockContinues.discard (Evaluates.append
      (Evaluates.lookup ((allKept "p" (by decide)).trans getP)) (found base values)
      (Evaluates.lookup reducedTotal) (found base values))
  refine ⟨reduced, ?_, reducedTotal, allKept⟩
  exact (BlockContinues.assign (name := "total") (Evaluates.constant (value := .int 0))).append
    ((BlockContinues.assign (name := "k") (Evaluates.constant (value := .int 1))).append
      (scanned.append (reducedRun.append appended)))

end solvers.«78».solve
