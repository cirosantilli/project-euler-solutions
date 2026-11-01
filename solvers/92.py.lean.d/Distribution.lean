import solvers.«92.py.lean.d».PrefixLoop
import solvers.«92.py.lean.d».Initialization

namespace solvers.«92».distribution
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev program := prefixLoop.program
abbrev body : List Stmt :=
  [.assign [.name "counts"] initialization.expression, prefixLoop.loop]

theorem source :
    ((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 1).take 2 = body := rfl

/-- The completed source block leaves an exact distribution in its shared counts dictionary. -/
def Post (bound : Nat) (base : Mutable.Heap) (following : List Stmt) (stack : List HeapCalls.Frame)
    (state : HeapCalls.State) : Prop :=
  ∃ locals heap address entries extra,
    state = ⟨.block following locals, stack, heap⟩ ∧ heap = base ++ extra ∧
    locals "counts" = some (.ref address) ∧ table.Found heap address entries ∧ table.Keys heap entries ∧
    Prefix.Exact bound (table.view entries)

def Verified (bound : Nat) (base : Mutable.Heap) (globals : Mutable.Globals) : Prop :=
  ∀ (before : HeapCalls.Env), before "limit_exclusive" = some (table.number (bound + 1)) →
    ∀ following anchors stack,
      HeapMemoCalls.RunsRel program initialization.spec anchors
        ⟨.block (body ++ following) before, stack, base⟩ (Post bound base following stack) globals

/-- Initialization and every decimal-prefix expansion preserve the compiler's pooled tuple identity. -/
theorem verified (bound : Nat) (base : Mutable.Heap) (globals : Mutable.Globals) :
    Verified bound base globals := by
  intro before limit following anchors stack
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.expr initialization.expression before,
      .assign "counts" (prefixLoop.loop :: following) before :: stack, base⟩)
    (by trivial) rfl
  apply (initialization.result base before globals anchors _).bind
  rintro state ⟨value, middle, rfl, keyAddress, address, firstExtra, rfl, extension, found, keys⟩
  let initial := setLocal before "counts" (.ref address)
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.block (prefixLoop.loop :: following) initial, stack, middle⟩)
    (by trivial) rfl
  obtain ⟨entries, extra, finalAddress, finalLocals, ran, finalFound, finalKeys, exact, finalCounts, _⟩ :=
    prefixLoop.result middle initial globals bound address (initialization.entries keyAddress)
      (by simpa [initial, setLocal] using limit) (by simp [initial, setLocal]) found keys
      (initialization.exact keyAddress)
  have disjoint : HeapMemoCalls.CacheDisjoint program program := by
    intro name impossible
    have empty : program.memoizedFunctions = [] := rfl
    rw [empty] at impossible
    cases impossible
  have lifted := HeapMemoCalls.Runs.from_heap (HeapCalls.Extends.refl program) disjoint
    (ran following anchors stack) (initialization.spec_preserves_append middle extra)
  apply lifted.consequence
  intro final same
  subst final
  refine ⟨finalLocals, middle ++ extra, finalAddress, entries, firstExtra ++ extra,
    rfl, ?_, finalCounts, finalFound, finalKeys, exact⟩
  simp [extension, List.append_assoc]

end solvers.«92».distribution
