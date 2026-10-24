import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row10589_small : List Nat := []
theorem row10589_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10589 row10589_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10589_block_005 : List Nat := []
theorem row10589_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10589 row10589_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10589_block_006 : List Nat := []
def plan10589_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 17), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .second, .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .second, .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 29), (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .second, (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.firstFactor 37), (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 13), (.secondFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 29), .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row10589_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10589 row10589_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10589_block_006
  · decide
  · rfl

def row10589_block_007 : List Nat := [12611]
def plan10589_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 29), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .second, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 23), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 31), .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .second, .second, .first, (.firstFactor 17), .first, (.secondFactor 31), .first, (.firstFactor 31), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 13), .first, .first, (.firstFactor 29), (.firstFactor 13), .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), (.secondFactor 37), .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 29), (.secondFactor 31), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10589_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10589 row10589_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10589_block_007
  · decide
  · rfl

def row10589_block_008 : List Nat := [15149]
def plan10589_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .second, .first, (.secondFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 29), (.firstFactor 19), .first, .first, .second, .first, (.firstFactor 37), (.firstFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 23), .second, .first, (.secondFactor 31), (.firstFactor 7), .first, .possible, (.secondFactor 7), (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 37), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10589_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10589 row10589_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10589_block_008
  · decide
  · rfl

def row10589_block_009 : List Nat := []
theorem row10589_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10589 row10589_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10589_block_010 : List Nat := []
theorem row10589_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10589 row10589_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10589_block_011 : List Nat := []
theorem row10589_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10589 row10589_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10589_block_012 : List Nat := []
theorem row10589_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10589 row10589_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10589_block_013 : List Nat := []
theorem row10589_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10589 row10589_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10589_large_left_left : List Nat := row10589_small ++ row10589_block_005
theorem row10589_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10589 row10589_large_left_left :=
  Budget.RowUpperBound.branch row10589_small_covers row10589_block_005_covers

def row10589_large_left_right_right : List Nat := row10589_block_007 ++ row10589_block_008
theorem row10589_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10589 row10589_large_left_right_right :=
  Budget.RowUpperBound.branch row10589_block_007_covers row10589_block_008_covers

def row10589_large_left_right : List Nat := row10589_block_006 ++ row10589_large_left_right_right
theorem row10589_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10589 row10589_large_left_right :=
  Budget.RowUpperBound.branch row10589_block_006_covers row10589_large_left_right_right_covers

def row10589_large_left : List Nat := row10589_large_left_left ++ row10589_large_left_right
theorem row10589_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10589 row10589_large_left :=
  Budget.RowUpperBound.branch row10589_large_left_left_covers row10589_large_left_right_covers

def row10589_large_right_left : List Nat := row10589_block_009 ++ row10589_block_010
theorem row10589_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10589 row10589_large_right_left :=
  Budget.RowUpperBound.branch row10589_block_009_covers row10589_block_010_covers

def row10589_large_right_right_right : List Nat := row10589_block_012 ++ row10589_block_013
theorem row10589_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10589 row10589_large_right_right_right :=
  Budget.RowUpperBound.branch row10589_block_012_covers row10589_block_013_covers

def row10589_large_right_right : List Nat := row10589_block_011 ++ row10589_large_right_right_right
theorem row10589_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10589 row10589_large_right_right :=
  Budget.RowUpperBound.branch row10589_block_011_covers row10589_large_right_right_right_covers

def row10589_large_right : List Nat := row10589_large_right_left ++ row10589_large_right_right
theorem row10589_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10589 row10589_large_right :=
  Budget.RowUpperBound.branch row10589_large_right_left_covers row10589_large_right_right_covers

def row10589_large : List Nat := row10589_large_left ++ row10589_large_right
theorem row10589_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10589 row10589_large :=
  Budget.RowUpperBound.branch row10589_large_left_covers row10589_large_right_covers

def row10589 : List Nat := row10589_large
theorem row10589_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10589 row10589 := row10589_large_covers

def row10597_small : List Nat := []
theorem row10597_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10597 row10597_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10597_block_005 : List Nat := []
theorem row10597_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10597 row10597_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10597_block_006 : List Nat := [10711, 11197, 11239, 11311, 11383, 11971]
def plan10597_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .second, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 37), (.firstFactor 11), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 13), .first, .possible, .first, .possible, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .second, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, .possible, .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 11), (.secondFactor 17), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .second, (.secondFactor 13), (.firstFactor 23), .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 37), .first, .first, (.secondFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row10597_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10597 row10597_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10597_block_006
  · decide
  · rfl

def row10597_block_007 : List Nat := []
def plan10597_block_007 : List Budget.Choice := [(.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.secondFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .second, .second, .first, (.secondFactor 7), (.secondFactor 37), .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .second, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .second, (.firstFactor 17), .first, (.secondFactor 17), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, .second, .first, (.firstFactor 11), .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 37), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 19), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 29), (.firstFactor 29), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, .second, (.secondFactor 29), (.firstFactor 11), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .second, .second, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row10597_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10597 row10597_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10597_block_007
  · decide
  · rfl

def row10597_block_008 : List Nat := []
def plan10597_block_008 : List Budget.Choice := [(.secondFactor 7), .first, .second, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .second, .first, (.secondFactor 17), (.secondFactor 13), .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.secondFactor 23), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 37), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10597_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10597 row10597_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10597_block_008
  · decide
  · rfl

def row10597_block_009 : List Nat := []
theorem row10597_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10597 row10597_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10597_block_010 : List Nat := []
theorem row10597_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10597 row10597_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10597_block_011 : List Nat := []
theorem row10597_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10597 row10597_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10597_block_012 : List Nat := []
theorem row10597_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10597 row10597_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10597_block_013 : List Nat := []
theorem row10597_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10597 row10597_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10597_large_left_left : List Nat := row10597_small ++ row10597_block_005
theorem row10597_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10597 row10597_large_left_left :=
  Budget.RowUpperBound.branch row10597_small_covers row10597_block_005_covers

def row10597_large_left_right_right : List Nat := row10597_block_007 ++ row10597_block_008
theorem row10597_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10597 row10597_large_left_right_right :=
  Budget.RowUpperBound.branch row10597_block_007_covers row10597_block_008_covers

def row10597_large_left_right : List Nat := row10597_block_006 ++ row10597_large_left_right_right
theorem row10597_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10597 row10597_large_left_right :=
  Budget.RowUpperBound.branch row10597_block_006_covers row10597_large_left_right_right_covers

def row10597_large_left : List Nat := row10597_large_left_left ++ row10597_large_left_right
theorem row10597_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10597 row10597_large_left :=
  Budget.RowUpperBound.branch row10597_large_left_left_covers row10597_large_left_right_covers

def row10597_large_right_left : List Nat := row10597_block_009 ++ row10597_block_010
theorem row10597_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10597 row10597_large_right_left :=
  Budget.RowUpperBound.branch row10597_block_009_covers row10597_block_010_covers

def row10597_large_right_right_right : List Nat := row10597_block_012 ++ row10597_block_013
theorem row10597_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10597 row10597_large_right_right_right :=
  Budget.RowUpperBound.branch row10597_block_012_covers row10597_block_013_covers

def row10597_large_right_right : List Nat := row10597_block_011 ++ row10597_large_right_right_right
theorem row10597_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10597 row10597_large_right_right :=
  Budget.RowUpperBound.branch row10597_block_011_covers row10597_large_right_right_right_covers

def row10597_large_right : List Nat := row10597_large_right_left ++ row10597_large_right_right
theorem row10597_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10597 row10597_large_right :=
  Budget.RowUpperBound.branch row10597_large_right_left_covers row10597_large_right_right_covers

def row10597_large : List Nat := row10597_large_left ++ row10597_large_right
theorem row10597_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10597 row10597_large :=
  Budget.RowUpperBound.branch row10597_large_left_covers row10597_large_right_covers

def row10597 : List Nat := row10597_large
theorem row10597_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10597 row10597 := row10597_large_covers

def row10601_small : List Nat := []
theorem row10601_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10601 row10601_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10601_block_005 : List Nat := []
theorem row10601_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10601 row10601_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10601_block_006 : List Nat := [11399, 11903]
def plan10601_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 31), (.secondFactor 13), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 17), (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 17), .first, .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .possible, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.secondFactor 19), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .possible, (.secondFactor 31), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 11), (.secondFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 31), .first, .first, (.secondFactor 29), (.secondFactor 13), .first, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 17), .first, .first]
set_option maxRecDepth 2048 in
theorem row10601_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10601 row10601_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10601_block_006
  · decide
  · rfl

def row10601_block_007 : List Nat := [12917]
def plan10601_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .second, (.secondFactor 29), (.firstFactor 17), .first, .first, (.firstFactor 23), .second, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 31), .first, (.secondFactor 29), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .possible, .first, (.firstFactor 19), (.firstFactor 23), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .second, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 23), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), (.firstFactor 13), (.secondFactor 37), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 37), (.secondFactor 17), .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 31), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 13), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 19), .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row10601_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10601 row10601_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10601_block_007
  · decide
  · rfl

def row10601_block_008 : List Nat := [15383]
def plan10601_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, (.secondFactor 29), .first, .first, (.secondFactor 13), .first, (.firstFactor 23), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 31), (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10601_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10601 row10601_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10601_block_008
  · decide
  · rfl

def row10601_block_009 : List Nat := []
theorem row10601_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10601 row10601_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10601_block_010 : List Nat := []
theorem row10601_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10601 row10601_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10601_block_011 : List Nat := []
theorem row10601_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10601 row10601_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10601_block_012 : List Nat := []
theorem row10601_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10601 row10601_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10601_block_013 : List Nat := []
theorem row10601_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10601 row10601_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10601_large_left_left : List Nat := row10601_small ++ row10601_block_005
theorem row10601_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10601 row10601_large_left_left :=
  Budget.RowUpperBound.branch row10601_small_covers row10601_block_005_covers

def row10601_large_left_right_right : List Nat := row10601_block_007 ++ row10601_block_008
theorem row10601_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10601 row10601_large_left_right_right :=
  Budget.RowUpperBound.branch row10601_block_007_covers row10601_block_008_covers

def row10601_large_left_right : List Nat := row10601_block_006 ++ row10601_large_left_right_right
theorem row10601_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10601 row10601_large_left_right :=
  Budget.RowUpperBound.branch row10601_block_006_covers row10601_large_left_right_right_covers

def row10601_large_left : List Nat := row10601_large_left_left ++ row10601_large_left_right
theorem row10601_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10601 row10601_large_left :=
  Budget.RowUpperBound.branch row10601_large_left_left_covers row10601_large_left_right_covers

def row10601_large_right_left : List Nat := row10601_block_009 ++ row10601_block_010
theorem row10601_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10601 row10601_large_right_left :=
  Budget.RowUpperBound.branch row10601_block_009_covers row10601_block_010_covers

def row10601_large_right_right_right : List Nat := row10601_block_012 ++ row10601_block_013
theorem row10601_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10601 row10601_large_right_right_right :=
  Budget.RowUpperBound.branch row10601_block_012_covers row10601_block_013_covers

def row10601_large_right_right : List Nat := row10601_block_011 ++ row10601_large_right_right_right
theorem row10601_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10601 row10601_large_right_right :=
  Budget.RowUpperBound.branch row10601_block_011_covers row10601_large_right_right_right_covers

def row10601_large_right : List Nat := row10601_large_right_left ++ row10601_large_right_right
theorem row10601_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10601 row10601_large_right :=
  Budget.RowUpperBound.branch row10601_large_right_left_covers row10601_large_right_right_covers

def row10601_large : List Nat := row10601_large_left ++ row10601_large_right
theorem row10601_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10601 row10601_large :=
  Budget.RowUpperBound.branch row10601_large_left_covers row10601_large_right_covers

def row10601 : List Nat := row10601_large
theorem row10601_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10601 row10601 := row10601_large_covers

def row10607_small : List Nat := []
theorem row10607_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 10607 row10607_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row10607_block_005 : List Nat := []
theorem row10607_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 10607 row10607_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row10607_block_006 : List Nat := []
def plan10607_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 37), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.secondFactor 29), .first, (.firstFactor 17), .first, .first, .first, .second, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 31), (.secondFactor 29), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 13), .first, .second, .first, .first, (.firstFactor 11), (.secondFactor 23), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 23)]
set_option maxRecDepth 2048 in
theorem row10607_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 10607 row10607_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan10607_block_006
  · decide
  · rfl

def row10607_block_007 : List Nat := [14087]
def plan10607_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 31), (.firstFactor 7), (.secondFactor 13), (.firstFactor 17), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .second, (.firstFactor 11), .first, (.secondFactor 7), .second, (.secondFactor 29), (.secondFactor 37), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .second, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row10607_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 10607 row10607_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan10607_block_007
  · decide
  · rfl

def row10607_block_008 : List Nat := [14741, 15101, 15137]
def plan10607_block_008 : List Budget.Choice := [.first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .second, .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, .possible, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 19), .second, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .possible, (.firstFactor 29), .first, (.firstFactor 7), .possible, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 19), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row10607_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 10607 row10607_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan10607_block_008
  · decide
  · rfl

def row10607_block_009 : List Nat := []
theorem row10607_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 10607 row10607_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row10607_block_010 : List Nat := []
theorem row10607_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 10607 row10607_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row10607_block_011 : List Nat := []
theorem row10607_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 10607 row10607_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row10607_block_012 : List Nat := []
theorem row10607_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 10607 row10607_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row10607_block_013 : List Nat := []
theorem row10607_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 10607 row10607_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row10607_large_left_left : List Nat := row10607_small ++ row10607_block_005
theorem row10607_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 10607 row10607_large_left_left :=
  Budget.RowUpperBound.branch row10607_small_covers row10607_block_005_covers

def row10607_large_left_right_right : List Nat := row10607_block_007 ++ row10607_block_008
theorem row10607_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 10607 row10607_large_left_right_right :=
  Budget.RowUpperBound.branch row10607_block_007_covers row10607_block_008_covers

def row10607_large_left_right : List Nat := row10607_block_006 ++ row10607_large_left_right_right
theorem row10607_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 10607 row10607_large_left_right :=
  Budget.RowUpperBound.branch row10607_block_006_covers row10607_large_left_right_right_covers

def row10607_large_left : List Nat := row10607_large_left_left ++ row10607_large_left_right
theorem row10607_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 10607 row10607_large_left :=
  Budget.RowUpperBound.branch row10607_large_left_left_covers row10607_large_left_right_covers

def row10607_large_right_left : List Nat := row10607_block_009 ++ row10607_block_010
theorem row10607_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 10607 row10607_large_right_left :=
  Budget.RowUpperBound.branch row10607_block_009_covers row10607_block_010_covers

def row10607_large_right_right_right : List Nat := row10607_block_012 ++ row10607_block_013
theorem row10607_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 10607 row10607_large_right_right_right :=
  Budget.RowUpperBound.branch row10607_block_012_covers row10607_block_013_covers

def row10607_large_right_right : List Nat := row10607_block_011 ++ row10607_large_right_right_right
theorem row10607_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 10607 row10607_large_right_right :=
  Budget.RowUpperBound.branch row10607_block_011_covers row10607_large_right_right_right_covers

def row10607_large_right : List Nat := row10607_large_right_left ++ row10607_large_right_right
theorem row10607_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 10607 row10607_large_right :=
  Budget.RowUpperBound.branch row10607_large_right_left_covers row10607_large_right_right_covers

def row10607_large : List Nat := row10607_large_left ++ row10607_large_right
theorem row10607_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 10607 row10607_large :=
  Budget.RowUpperBound.branch row10607_large_left_covers row10607_large_right_covers

def row10607 : List Nat := row10607_large
theorem row10607_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 10607 row10607 := row10607_large_covers

end ProjectEulerSolutions.PrimePairLarge
