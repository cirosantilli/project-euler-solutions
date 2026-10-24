import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8629_small : List Nat := []
theorem row8629_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8629 row8629_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8629_block_005 : List Nat := [10267]
def plan8629_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 11), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 17), (.secondFactor 19), (.firstFactor 11), (.firstFactor 13), (.secondFactor 13), (.firstFactor 31), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .second, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 31), (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 23), .first, .first, .first, (.firstFactor 17), (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 13), .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .possible, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8629_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8629 row8629_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8629_block_005
  · decide
  · rfl

def row8629_block_006 : List Nat := [11059]
def plan8629_block_006 : List Budget.Choice := [.first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .second, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .second, .first, .possible, .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .second, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 13), .second, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .second, .first, .first, .first, .first, .second, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 37), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .second, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 11), (.firstFactor 23), (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 37), .first]
set_option maxRecDepth 2048 in
theorem row8629_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8629 row8629_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8629_block_006
  · decide
  · rfl

def row8629_block_007 : List Nat := [12577, 12703, 13267, 14029]
def plan8629_block_007 : List Budget.Choice := [(.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, (.firstFactor 13), (.firstFactor 7), (.secondFactor 11), .first, .first, .second, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .second, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .possible, (.firstFactor 19), (.firstFactor 37), (.secondFactor 19), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .second, (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 17), .first, (.firstFactor 31), .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row8629_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8629 row8629_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8629_block_007
  · decide
  · rfl

def row8629_block_008 : List Nat := [14869, 15451, 15583]
def plan8629_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 23), .first, .possible, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 37), .first, .first, .second, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 23), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .second, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .possible, (.secondFactor 13), (.secondFactor 7), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 29), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 37), .first, .second, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 13), .first, .first, (.secondFactor 19), .first, .first, .second, .second, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8629_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8629 row8629_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8629_block_008
  · decide
  · rfl

def row8629_block_009 : List Nat := [16633, 16981]
def plan8629_block_009 : List Budget.Choice := [.first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .second, .first, .first, .first, .possible, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 31), (.firstFactor 23), (.secondFactor 13), .first, .first, .first, .second, .first, .possible, .second, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), .second, .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8629_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8629 row8629_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8629_block_009
  · decide
  · rfl

def row8629_block_010 : List Nat := []
theorem row8629_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8629 row8629_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8629_block_011 : List Nat := []
theorem row8629_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8629 row8629_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8629_block_012 : List Nat := []
theorem row8629_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8629 row8629_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8629_block_013 : List Nat := []
theorem row8629_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8629 row8629_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8629_large_left_left : List Nat := row8629_small ++ row8629_block_005
theorem row8629_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8629 row8629_large_left_left :=
  Budget.RowUpperBound.branch row8629_small_covers row8629_block_005_covers

def row8629_large_left_right_right : List Nat := row8629_block_007 ++ row8629_block_008
theorem row8629_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8629 row8629_large_left_right_right :=
  Budget.RowUpperBound.branch row8629_block_007_covers row8629_block_008_covers

def row8629_large_left_right : List Nat := row8629_block_006 ++ row8629_large_left_right_right
theorem row8629_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8629 row8629_large_left_right :=
  Budget.RowUpperBound.branch row8629_block_006_covers row8629_large_left_right_right_covers

def row8629_large_left : List Nat := row8629_large_left_left ++ row8629_large_left_right
theorem row8629_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8629 row8629_large_left :=
  Budget.RowUpperBound.branch row8629_large_left_left_covers row8629_large_left_right_covers

def row8629_large_right_left : List Nat := row8629_block_009 ++ row8629_block_010
theorem row8629_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8629 row8629_large_right_left :=
  Budget.RowUpperBound.branch row8629_block_009_covers row8629_block_010_covers

def row8629_large_right_right_right : List Nat := row8629_block_012 ++ row8629_block_013
theorem row8629_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8629 row8629_large_right_right_right :=
  Budget.RowUpperBound.branch row8629_block_012_covers row8629_block_013_covers

def row8629_large_right_right : List Nat := row8629_block_011 ++ row8629_large_right_right_right
theorem row8629_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8629 row8629_large_right_right :=
  Budget.RowUpperBound.branch row8629_block_011_covers row8629_large_right_right_right_covers

def row8629_large_right : List Nat := row8629_large_right_left ++ row8629_large_right_right
theorem row8629_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8629 row8629_large_right :=
  Budget.RowUpperBound.branch row8629_large_right_left_covers row8629_large_right_right_covers

def row8629_large : List Nat := row8629_large_left ++ row8629_large_right
theorem row8629_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8629 row8629_large :=
  Budget.RowUpperBound.branch row8629_large_left_covers row8629_large_right_covers

def row8629 : List Nat := row8629_large
theorem row8629_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8629 row8629 := row8629_large_covers

def row8641_small : List Nat := []
theorem row8641_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8641 row8641_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8641_block_005 : List Nat := [8719, 9649]
def plan8641_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .second, .possible, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 37), .first, .first, (.firstFactor 29), (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 31), (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 29), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 23), .first, .first, (.firstFactor 19), .second, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 23), (.secondFactor 13), .possible, (.firstFactor 31), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 37), .first, (.secondFactor 13), .first, .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row8641_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8641 row8641_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8641_block_005
  · decide
  · rfl

def row8641_block_006 : List Nat := [10531, 10729, 11491, 12157]
def plan8641_block_006 : List Budget.Choice := [.first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .possible, .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, .first, (.firstFactor 7), .second, (.secondFactor 31), .first, (.secondFactor 11), .first, (.firstFactor 31), .first, .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .possible, .first, .second, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .second, (.secondFactor 13), .first, (.firstFactor 13), (.secondFactor 23), (.firstFactor 23), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 29), .first, .second, .first]
set_option maxRecDepth 2048 in
theorem row8641_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8641 row8641_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8641_block_006
  · decide
  · rfl

def row8641_block_007 : List Nat := [12511, 13267, 13963]
def plan8641_block_007 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .possible, (.secondFactor 7), .first, .first, (.firstFactor 37), (.secondFactor 19), (.firstFactor 23), .first, .second, (.secondFactor 17), (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 7), .second, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 13), (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, (.secondFactor 11), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 11), .first, .possible, .first, (.secondFactor 17), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 13), (.secondFactor 23), .first, .second, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 19), .second, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8641_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8641 row8641_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8641_block_007
  · decide
  · rfl

def row8641_block_008 : List Nat := [14821, 15061, 15271, 15787, 16249]
def plan8641_block_008 : List Budget.Choice := [(.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .possible, (.firstFactor 29), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 37), (.secondFactor 23), .first, (.firstFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 19), .second, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .possible, (.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 19), (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.secondFactor 19), .first, .first, .first, .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8641_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8641 row8641_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8641_block_008
  · decide
  · rfl

def row8641_block_009 : List Nat := [16657, 17299]
def plan8641_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .possible, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 19), (.secondFactor 7), .first, .first, .second, .first, .first, (.secondFactor 31), (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), .possible, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 23), .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8641_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8641 row8641_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8641_block_009
  · decide
  · rfl

def row8641_block_010 : List Nat := []
theorem row8641_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8641 row8641_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8641_block_011 : List Nat := []
theorem row8641_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8641 row8641_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8641_block_012 : List Nat := []
theorem row8641_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8641 row8641_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8641_block_013 : List Nat := []
theorem row8641_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8641 row8641_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8641_large_left_left : List Nat := row8641_small ++ row8641_block_005
theorem row8641_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8641 row8641_large_left_left :=
  Budget.RowUpperBound.branch row8641_small_covers row8641_block_005_covers

def row8641_large_left_right_right : List Nat := row8641_block_007 ++ row8641_block_008
theorem row8641_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8641 row8641_large_left_right_right :=
  Budget.RowUpperBound.branch row8641_block_007_covers row8641_block_008_covers

def row8641_large_left_right : List Nat := row8641_block_006 ++ row8641_large_left_right_right
theorem row8641_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8641 row8641_large_left_right :=
  Budget.RowUpperBound.branch row8641_block_006_covers row8641_large_left_right_right_covers

def row8641_large_left : List Nat := row8641_large_left_left ++ row8641_large_left_right
theorem row8641_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8641 row8641_large_left :=
  Budget.RowUpperBound.branch row8641_large_left_left_covers row8641_large_left_right_covers

def row8641_large_right_left : List Nat := row8641_block_009 ++ row8641_block_010
theorem row8641_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8641 row8641_large_right_left :=
  Budget.RowUpperBound.branch row8641_block_009_covers row8641_block_010_covers

def row8641_large_right_right_right : List Nat := row8641_block_012 ++ row8641_block_013
theorem row8641_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8641 row8641_large_right_right_right :=
  Budget.RowUpperBound.branch row8641_block_012_covers row8641_block_013_covers

def row8641_large_right_right : List Nat := row8641_block_011 ++ row8641_large_right_right_right
theorem row8641_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8641 row8641_large_right_right :=
  Budget.RowUpperBound.branch row8641_block_011_covers row8641_large_right_right_right_covers

def row8641_large_right : List Nat := row8641_large_right_left ++ row8641_large_right_right
theorem row8641_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8641 row8641_large_right :=
  Budget.RowUpperBound.branch row8641_large_right_left_covers row8641_large_right_right_covers

def row8641_large : List Nat := row8641_large_left ++ row8641_large_right
theorem row8641_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8641 row8641_large :=
  Budget.RowUpperBound.branch row8641_large_left_covers row8641_large_right_covers

def row8641 : List Nat := row8641_large
theorem row8641_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8641 row8641 := row8641_large_covers

def row8647_small : List Nat := []
theorem row8647_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8647 row8647_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8647_block_005 : List Nat := [8713, 8929, 9007, 9091, 9403]
def plan8647_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 37), .second, .first, .first, .first, .second, (.firstFactor 17), .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .possible, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .possible, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 31), (.firstFactor 17), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 37), .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.secondFactor 37), (.secondFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, .second, (.firstFactor 7), .second, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 17), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 23)]
set_option maxRecDepth 2048 in
theorem row8647_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8647 row8647_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8647_block_005
  · decide
  · rfl

def row8647_block_006 : List Nat := [11503, 11863]
def plan8647_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .second, (.secondFactor 17), .first, (.secondFactor 29), .first, (.firstFactor 17), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .possible, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 13), (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .second, .possible, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 19), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row8647_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8647 row8647_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8647_block_006
  · decide
  · rfl

def row8647_block_007 : List Nat := [13159, 13417, 13963]
def plan8647_block_007 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 37), (.secondFactor 7), (.firstFactor 23), (.secondFactor 23), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .second, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), (.secondFactor 11), .first, .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 31), .possible, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, .second, .first, (.secondFactor 13), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row8647_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8647 row8647_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8647_block_007
  · decide
  · rfl

def row8647_block_008 : List Nat := [15427, 16231, 16273]
def plan8647_block_008 : List Budget.Choice := [(.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 23), (.firstFactor 37), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 23), (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 13), (.secondFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 29), (.secondFactor 17), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 11), .first, .second, .first, (.firstFactor 37), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 17), .second, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 29), .first, (.secondFactor 31), .first, .first, .first, .first, .possible, .first, .first, .first, .possible, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 37), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8647_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8647 row8647_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8647_block_008
  · decide
  · rfl

def row8647_block_009 : List Nat := [16981]
def plan8647_block_009 : List Budget.Choice := [(.firstFactor 29), .first, .first, .first, (.secondFactor 31), (.firstFactor 11), (.firstFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .second, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .possible, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), (.firstFactor 37), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 31), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8647_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8647 row8647_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8647_block_009
  · decide
  · rfl

def row8647_block_010 : List Nat := []
theorem row8647_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8647 row8647_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8647_block_011 : List Nat := []
theorem row8647_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8647 row8647_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8647_block_012 : List Nat := []
theorem row8647_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8647 row8647_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8647_block_013 : List Nat := []
theorem row8647_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8647 row8647_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8647_large_left_left : List Nat := row8647_small ++ row8647_block_005
theorem row8647_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8647 row8647_large_left_left :=
  Budget.RowUpperBound.branch row8647_small_covers row8647_block_005_covers

def row8647_large_left_right_right : List Nat := row8647_block_007 ++ row8647_block_008
theorem row8647_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8647 row8647_large_left_right_right :=
  Budget.RowUpperBound.branch row8647_block_007_covers row8647_block_008_covers

def row8647_large_left_right : List Nat := row8647_block_006 ++ row8647_large_left_right_right
theorem row8647_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8647 row8647_large_left_right :=
  Budget.RowUpperBound.branch row8647_block_006_covers row8647_large_left_right_right_covers

def row8647_large_left : List Nat := row8647_large_left_left ++ row8647_large_left_right
theorem row8647_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8647 row8647_large_left :=
  Budget.RowUpperBound.branch row8647_large_left_left_covers row8647_large_left_right_covers

def row8647_large_right_left : List Nat := row8647_block_009 ++ row8647_block_010
theorem row8647_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8647 row8647_large_right_left :=
  Budget.RowUpperBound.branch row8647_block_009_covers row8647_block_010_covers

def row8647_large_right_right_right : List Nat := row8647_block_012 ++ row8647_block_013
theorem row8647_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8647 row8647_large_right_right_right :=
  Budget.RowUpperBound.branch row8647_block_012_covers row8647_block_013_covers

def row8647_large_right_right : List Nat := row8647_block_011 ++ row8647_large_right_right_right
theorem row8647_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8647 row8647_large_right_right :=
  Budget.RowUpperBound.branch row8647_block_011_covers row8647_large_right_right_right_covers

def row8647_large_right : List Nat := row8647_large_right_left ++ row8647_large_right_right
theorem row8647_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8647 row8647_large_right :=
  Budget.RowUpperBound.branch row8647_large_right_left_covers row8647_large_right_right_covers

def row8647_large : List Nat := row8647_large_left ++ row8647_large_right
theorem row8647_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8647 row8647_large :=
  Budget.RowUpperBound.branch row8647_large_left_covers row8647_large_right_covers

def row8647 : List Nat := row8647_large
theorem row8647_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8647 row8647 := row8647_large_covers

def row8663_small : List Nat := []
theorem row8663_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8663 row8663_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8663_block_005 : List Nat := [9461, 9629, 10337, 10343]
def plan8663_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .second, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 19), .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 31), .first, .possible, .first, .first, .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .second, .second, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .second, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), (.secondFactor 17), .first, (.firstFactor 11), .second, (.firstFactor 17), .first, .second, .first, (.firstFactor 13), (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .possible, .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row8663_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8663 row8663_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8663_block_005
  · decide
  · rfl

def row8663_block_006 : List Nat := [11909, 12203]
def plan8663_block_006 : List Budget.Choice := [.first, .first, (.secondFactor 17), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 37), .first, .first, .first, .second, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 31), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 23), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 17), (.secondFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .second, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .second, .possible, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 29), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 23), .second, .first, .second, (.secondFactor 19), (.firstFactor 31), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 17), .possible, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 37)]
set_option maxRecDepth 2048 in
theorem row8663_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8663 row8663_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8663_block_006
  · decide
  · rfl

def row8663_block_007 : List Nat := [12437, 13127]
def plan8663_block_007 : List Budget.Choice := [.first, .first, .possible, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 23), .second, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .second, .first, .possible, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 11), .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .second, .second, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .second, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), (.secondFactor 11), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8663_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8663 row8663_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8663_block_007
  · decide
  · rfl

def row8663_block_008 : List Nat := [14717, 15101, 15803]
def plan8663_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 19), .first, (.firstFactor 19), .second, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 7), .first, .possible, (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 13), (.firstFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.firstFactor 37), (.secondFactor 19), .first, .possible, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 29), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 31), (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 11), .first, .possible, .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 23), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 19), (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 31), .second, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8663_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8663 row8663_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8663_block_008
  · decide
  · rfl

def row8663_block_009 : List Nat := []
def plan8663_block_009 : List Budget.Choice := [.first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 17), (.secondFactor 11), .first, .first, .first, (.secondFactor 23), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 19), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8663_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8663 row8663_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8663_block_009
  · decide
  · rfl

def row8663_block_010 : List Nat := []
theorem row8663_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8663 row8663_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8663_block_011 : List Nat := []
theorem row8663_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8663 row8663_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8663_block_012 : List Nat := []
theorem row8663_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8663 row8663_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8663_block_013 : List Nat := []
theorem row8663_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8663 row8663_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8663_large_left_left : List Nat := row8663_small ++ row8663_block_005
theorem row8663_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8663 row8663_large_left_left :=
  Budget.RowUpperBound.branch row8663_small_covers row8663_block_005_covers

def row8663_large_left_right_right : List Nat := row8663_block_007 ++ row8663_block_008
theorem row8663_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8663 row8663_large_left_right_right :=
  Budget.RowUpperBound.branch row8663_block_007_covers row8663_block_008_covers

def row8663_large_left_right : List Nat := row8663_block_006 ++ row8663_large_left_right_right
theorem row8663_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8663 row8663_large_left_right :=
  Budget.RowUpperBound.branch row8663_block_006_covers row8663_large_left_right_right_covers

def row8663_large_left : List Nat := row8663_large_left_left ++ row8663_large_left_right
theorem row8663_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8663 row8663_large_left :=
  Budget.RowUpperBound.branch row8663_large_left_left_covers row8663_large_left_right_covers

def row8663_large_right_left : List Nat := row8663_block_009 ++ row8663_block_010
theorem row8663_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8663 row8663_large_right_left :=
  Budget.RowUpperBound.branch row8663_block_009_covers row8663_block_010_covers

def row8663_large_right_right_right : List Nat := row8663_block_012 ++ row8663_block_013
theorem row8663_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8663 row8663_large_right_right_right :=
  Budget.RowUpperBound.branch row8663_block_012_covers row8663_block_013_covers

def row8663_large_right_right : List Nat := row8663_block_011 ++ row8663_large_right_right_right
theorem row8663_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8663 row8663_large_right_right :=
  Budget.RowUpperBound.branch row8663_block_011_covers row8663_large_right_right_right_covers

def row8663_large_right : List Nat := row8663_large_right_left ++ row8663_large_right_right
theorem row8663_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8663 row8663_large_right :=
  Budget.RowUpperBound.branch row8663_large_right_left_covers row8663_large_right_right_covers

def row8663_large : List Nat := row8663_large_left ++ row8663_large_right
theorem row8663_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8663 row8663_large :=
  Budget.RowUpperBound.branch row8663_large_left_covers row8663_large_right_covers

def row8663 : List Nat := row8663_large
theorem row8663_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8663 row8663 := row8663_large_covers

end ProjectEulerSolutions.PrimePairLarge
