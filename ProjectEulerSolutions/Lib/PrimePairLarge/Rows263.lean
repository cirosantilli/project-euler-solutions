import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8431_small : List Nat := []
theorem row8431_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8431 row8431_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8431_block_005 : List Nat := [9013, 9049, 9649, 10369]
def plan8431_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 19), .second, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.secondFactor 17), (.firstFactor 37), (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .possible, .first, .first, .first, .possible, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 11), (.firstFactor 7), (.firstFactor 23), (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 37), (.firstFactor 31), .first, .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 17), .possible, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .second, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 17), .first, .first, (.secondFactor 23), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .possible]
set_option maxRecDepth 2048 in
theorem row8431_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8431 row8431_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8431_block_005
  · decide
  · rfl

def row8431_block_006 : List Nat := [11251]
def plan8431_block_006 : List Budget.Choice := [.first, .second, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .second, .first, (.secondFactor 19), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.secondFactor 13), (.secondFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .possible, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .second, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 37), .first, .second, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .second, .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8431_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8431 row8431_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8431_block_006
  · decide
  · rfl

def row8431_block_007 : List Nat := [12421, 12511, 12583, 13309]
def plan8431_block_007 : List Budget.Choice := [.possible, (.firstFactor 17), .first, (.firstFactor 37), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 13), .first, .first, (.firstFactor 23), (.secondFactor 13), .first, .first, .first, .possible, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .second, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), (.secondFactor 31), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 29), .first, (.secondFactor 17), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 29), (.secondFactor 23), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 31), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 37), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8431_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8431 row8431_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8431_block_007
  · decide
  · rfl

def row8431_block_008 : List Nat := [15193, 15913, 16111]
def plan8431_block_008 : List Budget.Choice := [.second, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .possible, (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 31), .first, (.firstFactor 17), .first, (.secondFactor 11), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 31), .first, (.firstFactor 31), (.secondFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, .second, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 19), .possible, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 7), .first, .second, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 13), .first, .first, (.secondFactor 11), (.secondFactor 13), (.secondFactor 7), (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8431_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8431 row8431_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8431_block_008
  · decide
  · rfl

def row8431_block_009 : List Nat := [16987, 17293, 17581]
def plan8431_block_009 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 23), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .possible, (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.firstFactor 13), (.secondFactor 17), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 29), (.firstFactor 7), .first, .first, .possible, .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8431_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8431 row8431_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8431_block_009
  · decide
  · rfl

def row8431_block_010 : List Nat := []
theorem row8431_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8431 row8431_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8431_block_011 : List Nat := []
theorem row8431_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8431 row8431_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8431_block_012 : List Nat := []
theorem row8431_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8431 row8431_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8431_block_013 : List Nat := []
theorem row8431_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8431 row8431_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8431_large_left_left : List Nat := row8431_small ++ row8431_block_005
theorem row8431_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8431 row8431_large_left_left :=
  Budget.RowUpperBound.branch row8431_small_covers row8431_block_005_covers

def row8431_large_left_right_right : List Nat := row8431_block_007 ++ row8431_block_008
theorem row8431_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8431 row8431_large_left_right_right :=
  Budget.RowUpperBound.branch row8431_block_007_covers row8431_block_008_covers

def row8431_large_left_right : List Nat := row8431_block_006 ++ row8431_large_left_right_right
theorem row8431_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8431 row8431_large_left_right :=
  Budget.RowUpperBound.branch row8431_block_006_covers row8431_large_left_right_right_covers

def row8431_large_left : List Nat := row8431_large_left_left ++ row8431_large_left_right
theorem row8431_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8431 row8431_large_left :=
  Budget.RowUpperBound.branch row8431_large_left_left_covers row8431_large_left_right_covers

def row8431_large_right_left : List Nat := row8431_block_009 ++ row8431_block_010
theorem row8431_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8431 row8431_large_right_left :=
  Budget.RowUpperBound.branch row8431_block_009_covers row8431_block_010_covers

def row8431_large_right_right_right : List Nat := row8431_block_012 ++ row8431_block_013
theorem row8431_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8431 row8431_large_right_right_right :=
  Budget.RowUpperBound.branch row8431_block_012_covers row8431_block_013_covers

def row8431_large_right_right : List Nat := row8431_block_011 ++ row8431_large_right_right_right
theorem row8431_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8431 row8431_large_right_right :=
  Budget.RowUpperBound.branch row8431_block_011_covers row8431_large_right_right_right_covers

def row8431_large_right : List Nat := row8431_large_right_left ++ row8431_large_right_right
theorem row8431_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8431 row8431_large_right :=
  Budget.RowUpperBound.branch row8431_large_right_left_covers row8431_large_right_right_covers

def row8431_large : List Nat := row8431_large_left ++ row8431_large_right
theorem row8431_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8431 row8431_large :=
  Budget.RowUpperBound.branch row8431_large_left_covers row8431_large_right_covers

def row8431 : List Nat := row8431_large
theorem row8431_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8431 row8431 := row8431_large_covers

def row8443_small : List Nat := []
theorem row8443_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8443 row8443_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8443_block_005 : List Nat := [9001, 9349, 10111]
def plan8443_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 29), (.firstFactor 19), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 17), .first, .second, .first, .first, .first, (.secondFactor 7), (.secondFactor 31), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 29), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .possible, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, (.secondFactor 23), (.firstFactor 11), (.firstFactor 7), .second, .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 31), .first, .second, .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .second, (.firstFactor 7), (.firstFactor 37), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8443_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8443 row8443_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8443_block_005
  · decide
  · rfl

def row8443_block_006 : List Nat := [10711, 11119, 11491, 11719]
def plan8443_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, .second, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .possible, (.secondFactor 13), (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .possible, (.secondFactor 29), (.firstFactor 11), .first, .second, .first, .second, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, .possible, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .possible, (.secondFactor 11), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.secondFactor 23), (.firstFactor 13), .first, (.firstFactor 29), .first]
set_option maxRecDepth 2048 in
theorem row8443_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8443 row8443_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8443_block_006
  · decide
  · rfl

def row8443_block_007 : List Nat := [13009, 13033, 13339, 13633, 14281, 14293]
def plan8443_block_007 : List Budget.Choice := [.first, .first, .first, (.firstFactor 23), (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 23), (.firstFactor 11), .first, .second, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 23), .first, .possible, .possible, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .second, .first, (.secondFactor 29), .first, (.firstFactor 11), .second, .first, .first, (.firstFactor 29), .possible, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), (.firstFactor 19), .first, .first, (.secondFactor 13), .first, .second, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .possible, .possible, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), (.secondFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 7), .second, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, (.firstFactor 11), (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row8443_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8443 row8443_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8443_block_007
  · decide
  · rfl

def row8443_block_008 : List Nat := [15649, 15877, 15991]
def plan8443_block_008 : List Budget.Choice := [.second, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, .second, (.firstFactor 13), .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 19), (.firstFactor 23), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, (.firstFactor 17), .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, (.firstFactor 13), .second, .possible, .first, .first, (.firstFactor 29), (.firstFactor 11), (.firstFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .possible, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .second, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 17), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8443_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8443 row8443_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8443_block_008
  · decide
  · rfl

def row8443_block_009 : List Nat := [16831]
def plan8443_block_009 : List Budget.Choice := [(.secondFactor 23), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.secondFactor 13), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 23), .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 23), (.secondFactor 19), (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 17), (.secondFactor 11), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 31), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8443_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8443 row8443_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8443_block_009
  · decide
  · rfl

def row8443_block_010 : List Nat := []
theorem row8443_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8443 row8443_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8443_block_011 : List Nat := []
theorem row8443_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8443 row8443_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8443_block_012 : List Nat := []
theorem row8443_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8443 row8443_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8443_block_013 : List Nat := []
theorem row8443_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8443 row8443_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8443_large_left_left : List Nat := row8443_small ++ row8443_block_005
theorem row8443_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8443 row8443_large_left_left :=
  Budget.RowUpperBound.branch row8443_small_covers row8443_block_005_covers

def row8443_large_left_right_right : List Nat := row8443_block_007 ++ row8443_block_008
theorem row8443_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8443 row8443_large_left_right_right :=
  Budget.RowUpperBound.branch row8443_block_007_covers row8443_block_008_covers

def row8443_large_left_right : List Nat := row8443_block_006 ++ row8443_large_left_right_right
theorem row8443_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8443 row8443_large_left_right :=
  Budget.RowUpperBound.branch row8443_block_006_covers row8443_large_left_right_right_covers

def row8443_large_left : List Nat := row8443_large_left_left ++ row8443_large_left_right
theorem row8443_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8443 row8443_large_left :=
  Budget.RowUpperBound.branch row8443_large_left_left_covers row8443_large_left_right_covers

def row8443_large_right_left : List Nat := row8443_block_009 ++ row8443_block_010
theorem row8443_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8443 row8443_large_right_left :=
  Budget.RowUpperBound.branch row8443_block_009_covers row8443_block_010_covers

def row8443_large_right_right_right : List Nat := row8443_block_012 ++ row8443_block_013
theorem row8443_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8443 row8443_large_right_right_right :=
  Budget.RowUpperBound.branch row8443_block_012_covers row8443_block_013_covers

def row8443_large_right_right : List Nat := row8443_block_011 ++ row8443_large_right_right_right
theorem row8443_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8443 row8443_large_right_right :=
  Budget.RowUpperBound.branch row8443_block_011_covers row8443_large_right_right_right_covers

def row8443_large_right : List Nat := row8443_large_right_left ++ row8443_large_right_right
theorem row8443_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8443 row8443_large_right :=
  Budget.RowUpperBound.branch row8443_large_right_left_covers row8443_large_right_right_covers

def row8443_large : List Nat := row8443_large_left ++ row8443_large_right
theorem row8443_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8443 row8443_large :=
  Budget.RowUpperBound.branch row8443_large_left_covers row8443_large_right_covers

def row8443 : List Nat := row8443_large
theorem row8443_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8443 row8443 := row8443_large_covers

def row8447_small : List Nat := []
theorem row8447_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8447 row8447_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8447_block_005 : List Nat := [8699, 9173, 9431, 10247]
def plan8447_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 19), .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .possible, .first, .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 31), (.firstFactor 19), .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 31), (.firstFactor 17), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 29), .first, (.firstFactor 7), (.secondFactor 13), .second, (.firstFactor 7), .first, .possible, (.secondFactor 11), (.firstFactor 23), .first, (.secondFactor 13), .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8447_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8447 row8447_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8447_block_005
  · decide
  · rfl

def row8447_block_006 : List Nat := [11717]
def plan8447_block_006 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 29), .first, (.secondFactor 11), .first, .first, .second, (.firstFactor 7), .second, .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 23), .first, .first, .second, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .second, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 31), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, (.secondFactor 19), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .second, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .second]
set_option maxRecDepth 2048 in
theorem row8447_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8447 row8447_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8447_block_006
  · decide
  · rfl

def row8447_block_007 : List Nat := [12437, 12917, 13691, 13763, 14243, 14303]
def plan8447_block_007 : List Budget.Choice := [.first, .first, .possible, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 29), .first, (.firstFactor 7), .possible, .first, (.firstFactor 37), (.secondFactor 23), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 23), .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 11), (.firstFactor 7), .first, (.secondFactor 19), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .possible, .first, (.firstFactor 17), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), .second, .first, .possible, (.secondFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 23), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 37), (.secondFactor 17), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 17), .first, .possible, (.secondFactor 7), .first, .first, .first, .possible, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8447_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8447 row8447_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8447_block_007
  · decide
  · rfl

def row8447_block_008 : List Nat := [14723, 14831, 14957, 15077, 15647, 15881, 16001, 16217]
def plan8447_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 13), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 19), .first, (.firstFactor 23), .possible, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, .first, .possible, .second, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 23), (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .second, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 31), .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .second, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .possible, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 11), .first, .first, .possible, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .possible, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row8447_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8447 row8447_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8447_block_008
  · decide
  · rfl

def row8447_block_009 : List Nat := [16673, 17099]
def plan8447_block_009 : List Budget.Choice := [.first, (.secondFactor 23), .second, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), (.secondFactor 7), (.secondFactor 13), (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 23), (.secondFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .possible, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.secondFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 29), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8447_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8447 row8447_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8447_block_009
  · decide
  · rfl

def row8447_block_010 : List Nat := []
theorem row8447_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8447 row8447_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8447_block_011 : List Nat := []
theorem row8447_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8447 row8447_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8447_block_012 : List Nat := []
theorem row8447_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8447 row8447_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8447_block_013 : List Nat := []
theorem row8447_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8447 row8447_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8447_large_left_left : List Nat := row8447_small ++ row8447_block_005
theorem row8447_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8447 row8447_large_left_left :=
  Budget.RowUpperBound.branch row8447_small_covers row8447_block_005_covers

def row8447_large_left_right_right : List Nat := row8447_block_007 ++ row8447_block_008
theorem row8447_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8447 row8447_large_left_right_right :=
  Budget.RowUpperBound.branch row8447_block_007_covers row8447_block_008_covers

def row8447_large_left_right : List Nat := row8447_block_006 ++ row8447_large_left_right_right
theorem row8447_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8447 row8447_large_left_right :=
  Budget.RowUpperBound.branch row8447_block_006_covers row8447_large_left_right_right_covers

def row8447_large_left : List Nat := row8447_large_left_left ++ row8447_large_left_right
theorem row8447_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8447 row8447_large_left :=
  Budget.RowUpperBound.branch row8447_large_left_left_covers row8447_large_left_right_covers

def row8447_large_right_left : List Nat := row8447_block_009 ++ row8447_block_010
theorem row8447_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8447 row8447_large_right_left :=
  Budget.RowUpperBound.branch row8447_block_009_covers row8447_block_010_covers

def row8447_large_right_right_right : List Nat := row8447_block_012 ++ row8447_block_013
theorem row8447_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8447 row8447_large_right_right_right :=
  Budget.RowUpperBound.branch row8447_block_012_covers row8447_block_013_covers

def row8447_large_right_right : List Nat := row8447_block_011 ++ row8447_large_right_right_right
theorem row8447_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8447 row8447_large_right_right :=
  Budget.RowUpperBound.branch row8447_block_011_covers row8447_large_right_right_right_covers

def row8447_large_right : List Nat := row8447_large_right_left ++ row8447_large_right_right
theorem row8447_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8447 row8447_large_right :=
  Budget.RowUpperBound.branch row8447_large_right_left_covers row8447_large_right_right_covers

def row8447_large : List Nat := row8447_large_left ++ row8447_large_right
theorem row8447_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8447 row8447_large :=
  Budget.RowUpperBound.branch row8447_large_left_covers row8447_large_right_covers

def row8447 : List Nat := row8447_large
theorem row8447_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8447 row8447 := row8447_large_covers

def row8461_small : List Nat := []
theorem row8461_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8461 row8461_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8461_block_005 : List Nat := [8893, 9091, 9817, 9949]
def plan8461_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.secondFactor 23), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 13), .possible, (.firstFactor 19), (.secondFactor 17), .second, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .second, .first, (.secondFactor 17), (.firstFactor 11), (.firstFactor 31), .first, .first, .first, (.firstFactor 13), (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .possible, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 29), .first, (.firstFactor 19), .second, .first, .first, (.firstFactor 11), .first, .possible, (.firstFactor 23), .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .second, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row8461_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8461 row8461_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8461_block_005
  · decide
  · rfl

def row8461_block_006 : List Nat := [10477, 10597, 12073, 12301]
def plan8461_block_006 : List Budget.Choice := [.first, (.secondFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 11), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 23), .first, .first, (.secondFactor 19), (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 31), .first, (.secondFactor 29), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 23), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .possible, (.secondFactor 31), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .possible, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8461_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8461 row8461_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8461_block_006
  · decide
  · rfl

def row8461_block_007 : List Nat := [13963]
def plan8461_block_007 : List Budget.Choice := [(.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .second, .first, (.firstFactor 29), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .second, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, (.secondFactor 13), (.secondFactor 7), (.firstFactor 29), .first, (.secondFactor 29), .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, (.secondFactor 31), (.firstFactor 11), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 19), .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row8461_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8461 row8461_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8461_block_007
  · decide
  · rfl

def row8461_block_008 : List Nat := [15139, 15289, 15349, 15727, 15913]
def plan8461_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.secondFactor 31), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 11), (.secondFactor 31), .first, .first, .first, .first, .first, .second, .first, .first, .possible, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 17), .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, .second, .first, .possible, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 37), (.secondFactor 7), (.firstFactor 7), .possible, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 29), .second, .first, .first, (.firstFactor 17), (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8461_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8461 row8461_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8461_block_008
  · decide
  · rfl

def row8461_block_009 : List Nat := []
def plan8461_block_009 : List Budget.Choice := [.second, .first, .first, .first, (.secondFactor 31), (.secondFactor 13), (.firstFactor 31), .second, .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 17), (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 29), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8461_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8461 row8461_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8461_block_009
  · decide
  · rfl

def row8461_block_010 : List Nat := []
theorem row8461_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8461 row8461_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8461_block_011 : List Nat := []
theorem row8461_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8461 row8461_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8461_block_012 : List Nat := []
theorem row8461_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8461 row8461_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8461_block_013 : List Nat := []
theorem row8461_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8461 row8461_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8461_large_left_left : List Nat := row8461_small ++ row8461_block_005
theorem row8461_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8461 row8461_large_left_left :=
  Budget.RowUpperBound.branch row8461_small_covers row8461_block_005_covers

def row8461_large_left_right_right : List Nat := row8461_block_007 ++ row8461_block_008
theorem row8461_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8461 row8461_large_left_right_right :=
  Budget.RowUpperBound.branch row8461_block_007_covers row8461_block_008_covers

def row8461_large_left_right : List Nat := row8461_block_006 ++ row8461_large_left_right_right
theorem row8461_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8461 row8461_large_left_right :=
  Budget.RowUpperBound.branch row8461_block_006_covers row8461_large_left_right_right_covers

def row8461_large_left : List Nat := row8461_large_left_left ++ row8461_large_left_right
theorem row8461_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8461 row8461_large_left :=
  Budget.RowUpperBound.branch row8461_large_left_left_covers row8461_large_left_right_covers

def row8461_large_right_left : List Nat := row8461_block_009 ++ row8461_block_010
theorem row8461_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8461 row8461_large_right_left :=
  Budget.RowUpperBound.branch row8461_block_009_covers row8461_block_010_covers

def row8461_large_right_right_right : List Nat := row8461_block_012 ++ row8461_block_013
theorem row8461_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8461 row8461_large_right_right_right :=
  Budget.RowUpperBound.branch row8461_block_012_covers row8461_block_013_covers

def row8461_large_right_right : List Nat := row8461_block_011 ++ row8461_large_right_right_right
theorem row8461_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8461 row8461_large_right_right :=
  Budget.RowUpperBound.branch row8461_block_011_covers row8461_large_right_right_right_covers

def row8461_large_right : List Nat := row8461_large_right_left ++ row8461_large_right_right
theorem row8461_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8461 row8461_large_right :=
  Budget.RowUpperBound.branch row8461_large_right_left_covers row8461_large_right_right_covers

def row8461_large : List Nat := row8461_large_left ++ row8461_large_right
theorem row8461_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8461 row8461_large :=
  Budget.RowUpperBound.branch row8461_large_left_covers row8461_large_right_covers

def row8461 : List Nat := row8461_large
theorem row8461_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8461 row8461 := row8461_large_covers

end ProjectEulerSolutions.PrimePairLarge
