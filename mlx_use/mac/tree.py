# --- START OF FILE mac_use/mac/tree.py ---
import asyncio

# --- START OF FILE mac_use/mac/actions.py ---
import logging
from typing import Callable, Dict, List, Optional

import Cocoa
from ApplicationServices import AXUIElementPerformAction, AXUIElementSetAttributeValue, kAXPressAction, kAXValueAttribute
from Foundation import NSString

from mlx_use.mac.element import MacElementNode

logger = logging.getLogger(__name__)

import Cocoa
import objc
from ApplicationServices import (
	AXError,
	AXUIElementCopyActionNames,
	AXUIElementCopyAttributeValue,
	AXUIElementCreateApplication,
	kAXChildrenAttribute,
	kAXDescriptionAttribute,
	kAXErrorAPIDisabled,
	kAXErrorAttributeUnsupported,
	kAXErrorCannotComplete,
	kAXErrorFailure,
	kAXErrorIllegalArgument,
	kAXErrorSuccess,
	kAXMainWindowAttribute,
	kAXRoleAttribute,
	kAXTitleAttribute,
	kAXValueAttribute,
	kAXWindowsAttribute,
)
from CoreFoundation import CFRunLoopAddSource, CFRunLoopGetCurrent, kCFRunLoopDefaultMode

from .element import MacElementNode

logger = logging.getLogger(__name__)

# Constant list of AX attributes for enhanced UI tree details 
AX_ATTRIBUTES = [
	"AXARIAAtomic",
	"AXARIALive",
	"AXARIARelevant",
	"AXActivationPoint",
	"AXAlternateUIVisible",
	"AXApplication",
	"AXBlockQuoteLevel",
	"AXButton",
	"AXCaretBrowsingEnabled",
	"AXCheckBox",
	"AXChildrenInNavigationOrder",
	"AXCloseButton",
	"AXCodeStyleGroup",
	"AXContainer",
	"AXContent",
	"AXContentList",
	"AXContents",
	"AXDOMClassList",
	"AXDOMIdentifier",
	"AXDescription",
	"AXEditableAncestor",
	"AXEdited",
	"AXElementBusy",
	"AXEmbeddedImageDescription",
	"AXEmptyGroup",
	"AXEnabled",
	"AXEndTextMarker",
	"AXFieldset",
	"AXFocusableAncestor",
	"AXFocused",
	"AXFrame",
	"AXFullScreen",
	"AXFullScreenButton",
	"AXGroup",
	"AXHasDocumentRoleAncestor",
	"AXHasPopup",
	"AXHasWebApplicationAncestor",
	"AXHeading",
	"AXHelp",
	"AXHighestEditableAncestor",
	"AXHorizontalOrientation",
	"AXHorizontalScrollBar",
	"AXIdentifier",
	"AXImage",
	"AXInlineText",
	"AXInsertionPointLineNumber",
	"AXInvalid",
	"AXLandmarkNavigation",
	"AXLandmarkRegion",
	"AXLanguage",
	"AXLayoutCount",
	"AXLink",
	"AXLinkRelationshipType",
	"AXLinkUIElements",
	"AXLinkedUIElements",
	"AXList",
	"AXListMarker",
	"AXLoaded",
	"AXLoadingProgress",
	"AXMain",
	"AXMaxValue",
	"AXMenuButton",
	"AXMinValue",
	"AXMinimizeButton",
	"AXMinimized",
	"AXModal",
	"AXNextContents",
	"AXNumberOfCharacters",
	"AXOrientation",
	"AXParent",
	"AXPath",
	"AXPlaceholderValue",
	"AXPopUpButton",
	"AXPosition",
	"AXPreventKeyboardDOMEventDispatch",
	"AXRadioButton",
	"AXRelativeFrame",
	"AXRequired",
	"AXRoleDescription",
	"AXScrollArea",
	"AXScrollBar",
	"AXSections",
	"AXSegment",
	"AXSelected",
	"AXSelectedChildren",
	"AXSelectedTextMarkerRange",
	"AXSelectedTextRange",
	"AXSize",
	"AXSplitGroup",
	"AXSplitter",
	"AXSplitters",
	"AXStandardWindow",
	"AXStartTextMarker",
	"AXStaticText",
	"AXSubrole",
	"AXTabButton",
	"AXTabGroup",
	"AXTabs",
	"AXTextArea",
	"AXTextField",
	"AXTextMarker",
	"AXTextMarkerRange",
	"AXTitle",
	"AXToggle",
	"AXToolbar",
	"AXTopLevelNavigator",
	"AXTopLevelUIElement",
	"AXUIElement",
	"AXUIElementCopyAttributeNames",
	"AXUIElementCreateApplication",
	"AXURL",
	"AXUnknown",
	"AXValue",
	"AXValueAutofillAvailable",
	"AXVerticalOrientation",
	"AXVerticalScrollBar",
	"AXVisibleCharacterRange",
	"AXVisibleChildren",
	"AXVisited",
	"AXWebArea",
	"AXWindow",
	"AXZoomButton",
]

class MacUITreeBuilder:
	def __init__(self):
		self.highlight_index = 0
		self._element_cache = {}
		self._observers = {}
		self._processed_elements = set()
		self._current_app_pid = None
		self.max_depth = 10
		self.max_children = 50

		# Define interactive actions we care about
		self.INTERACTIVE_ACTIONS = {
			'AXPress',            # Most buttons and clickable elements
			'AXShowMenu',         # Menu buttons
			'AXIncrement',        # Spinners/steppers
			'AXDecrement',
			'AXConfirm',         # Dialogs
			'AXCancel',
			'AXRaise',           # Windows
			'AXSetValue'         # Text fields/inputs
		}

		# Actions that require scrolling
		self.SCROLL_ACTIONS = {
			'AXScrollLeftByPage',
			'AXScrollRightByPage',
			'AXScrollUpByPage',
			'AXScrollDownByPage'
		}

	def _setup_observer(self, pid: int) -> bool:
		"""Setup accessibility observer for an application"""
		return True  #  Temporarily always return True

	def _get_attribute(self, element: 'AXUIElement', attribute: str) -> any:
		"""Safely get an accessibility attribute with error reporting"""
		try:
			error, value_ref = AXUIElementCopyAttributeValue(element, attribute, None)
			if error == kAXErrorSuccess:
				return value_ref
			elif error == kAXErrorAttributeUnsupported:
				logger.debug(f"Attribute '{attribute}' is not supported for this element.")
				return None
			else:
				logger.debug(f"Error getting attribute '{attribute}': {error}")
				return None
		except Exception as e:
			logger.debug(f"Exception getting attribute '{attribute}': {str(e)}")
			return None

	def _get_actions(self, element: 'AXUIElement') -> List[str]:
		"""Get available actions for an element with proper error handling"""
		try:
			error, actions = AXUIElementCopyActionNames(element, None)
			if error == kAXErrorSuccess and actions:
				# Convert NSArray to Python list
				return list(actions)
			return []
		except Exception as e:
			logger.debug(f'Error getting actions: {e}')
			return []

	def _is_interactive(self, element: 'AXUIElement', role: str, actions: List[str]) -> bool:
		"""Determine if an element is truly interactive based on its role and actions."""
		if not actions:
			return False

		# Check if element has any interactive actions
		has_interactive = any(action in self.INTERACTIVE_ACTIONS for action in actions)
		has_scroll = any(action in self.SCROLL_ACTIONS for action in actions)
		
		# Special handling for text input fields
		if 'AXSetValue' in actions:
			enabled = self._get_attribute(element, 'AXEnabled')
			return bool(enabled)

		# Special handling for buttons with AXPress
		if 'AXPress' in actions and role == 'AXButton':
			enabled = self._get_attribute(element, 'AXEnabled')
			return bool(enabled)

		return has_interactive or has_scroll

	async def _process_element(self, element: 'AXUIElement', pid: int, parent: Optional[MacElementNode] = None, depth: int = 0) -> Optional[MacElementNode]:
		"""Process a single UI element"""
		element_identifier = str(element)
		
		if element_identifier in self._processed_elements:
			return None

		self._processed_elements.add(element_identifier)

		try:
			role = self._get_attribute(element, kAXRoleAttribute)
			if not role:
				return None

			# Get all possible attributes and actions
			actions = self._get_actions(element)
			
			# Create node with enhanced attributes
			node = MacElementNode(
				role=role,
				identifier=element_identifier,
				attributes={},
				is_visible=True,
				parent=parent,
				app_pid=pid,
			)
			node._element = element

			# Store the actions in the node's attributes for reference
			if actions:
				node.attributes['actions'] = actions

			# Get basic attributes
			title = self._get_attribute(element, kAXTitleAttribute)
			value = self._get_attribute(element, kAXValueAttribute)
			description = self._get_attribute(element, kAXDescriptionAttribute)
			is_enabled = self._get_attribute(element, 'AXEnabled')
			
			# Additional useful attributes
			position = self._get_attribute(element, 'AXPosition')
			size = self._get_attribute(element, 'AXSize')
			subrole = self._get_attribute(element, 'AXSubrole')

			# Update node attributes
			if title:
				node.attributes['title'] = title
			if value:
				node.attributes['value'] = value
			if description:
				node.attributes['description'] = description
			if is_enabled is not None:
				node.is_visible = bool(is_enabled)
				node.attributes['enabled'] = bool(is_enabled)
			if position:
				node.attributes['position'] = position
			if size:
				node.attributes['size'] = size
			if subrole:
				node.attributes['subrole'] = subrole

			# Determine interactivity based on actions
			node.is_interactive = self._is_interactive(element, role, actions)
			
			if node.is_interactive:
				node.highlight_index = self.highlight_index
				self._element_cache[self.highlight_index] = node
				self.highlight_index += 1
				logger.debug(f'Added interactive element {role} with actions: {actions}')

			# Process children
			children_ref = self._get_attribute(element, kAXChildrenAttribute)
			if children_ref and depth < self.max_depth:
				try:
					children_list = list(children_ref)[:self.max_children]
					for child in children_list:
						child_node = await self._process_element(child, pid, node, depth + 1)
						if child_node:
							node.children.append(child_node)
				except Exception as e:
					logger.warning(f"Error processing children: {e}")

			return node

		except Exception as e:
			logger.error(f'Error processing element: {str(e)}')
			return None

	async def build_tree(self, pid: Optional[int] = None) -> Optional[MacElementNode]:
		"""Build UI tree for a specific application"""
		try:
			if pid is None and self._current_app_pid is None:
				logger.debug('No app is currently open - waiting for app to be launched')
				raise ValueError('No app is currently open')

			if pid is not None:
				self._current_app_pid = pid

				if not self._setup_observer(self._current_app_pid):
					logger.warning('Failed to setup accessibility observer')
					return None

			logger.debug(f'Creating AX element for pid {self._current_app_pid}')
			app_ref = AXUIElementCreateApplication(self._current_app_pid)

			logger.debug('Testing accessibility permissions (Role)...')
			error, role_attr = AXUIElementCopyAttributeValue(app_ref, kAXRoleAttribute, None)
			if error == kAXErrorSuccess:
				logger.debug(f'Successfully got role attribute: ({error}, {role_attr})')
			else:
				logger.error(f'Error getting role attribute: {error}')
				if error == kAXErrorAPIDisabled:
					logger.error('Accessibility is not enabled. Please enable it in System Settings.')
				return None

			root = MacElementNode(
				role='application',
				identifier=str(app_ref),
				attributes={},
				is_visible=True,
				app_pid=self._current_app_pid,
			)
			root._element = app_ref

			logger.debug('Trying to get the main window...')
			error, main_window_ref = AXUIElementCopyAttributeValue(app_ref, kAXMainWindowAttribute, None)
			if error != kAXErrorSuccess or not main_window_ref:
				logger.warning(f'Could not get main window (error: {error}), trying fallback attribute AXWindows')
				error, windows = AXUIElementCopyAttributeValue(app_ref, kAXWindowsAttribute, None)
				if error == kAXErrorSuccess and windows:
					try:
						windows_list = list(windows)
						if windows_list:
							main_window_ref = windows_list[0]
							logger.debug(f'Fallback: selected first window from AXWindows: {main_window_ref}')
						else:
							logger.warning("Fallback: AXWindows returned an empty list")
					except Exception as e:
						logger.error(f'Failed to iterate over AXWindows: {e}')
				else:
					logger.error(f'Fallback failed: could not get AXWindows (error: {error})')

			if main_window_ref:
				logger.debug(f'Found main window: {main_window_ref}')
				window_node = await self._process_element(main_window_ref, self._current_app_pid, root)
				if window_node:
					root.children.append(window_node)
			else:
				logger.error('Could not determine a main window for the application.')

			return root

		except Exception as e:
			if 'No app is currently open' not in str(e):
				logger.error(f'Error building tree: {str(e)}')
				import traceback
				traceback.print_exc()
			return None

	def cleanup(self):
		"""Cleanup observers"""
		pass  # Temporarily do nothing
