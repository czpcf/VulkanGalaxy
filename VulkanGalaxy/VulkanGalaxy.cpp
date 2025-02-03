#include <vulkan/vulkan.h>

#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <iostream>
#include <vector>
#include <set>
#include <optional>
#include "utils.hpp"

#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif

const uint32_t WINDOW_WIDTH = 800;
const uint32_t WINDOW_HEIGHT = 600;
const std::string APP_NAME = "VulkanGalaxy";
const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

struct QueueFamilyIndices {
	std::optional<uint32_t> uniformFamily; // uniform = graphics + compute
	std::optional<uint32_t> presentFamily;

	bool isComplete() const {
		return uniformFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapchainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

static std::vector<const char*> getRequiredExtensions();
static bool isDeviceSuitable(VkPhysicalDevice, VkSurfaceKHR&);
static QueueFamilyIndices findQueueFamilyProperties(VkPhysicalDevice, VkSurfaceKHR&);
static SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice, VkSurfaceKHR&);


class App {
public:
	App() {}
	void run() {
		initWindow();
		initVulkan();
		std::cerr << "app intialized" << std::endl;
		mainLoop();
		cleanup();
	}
private:
	GLFWwindow* window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;
	VkDevice device;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	QueueFamilyIndices indices;
	SwapchainSupportDetails details;

	// initialize the window using glfw
	void initWindow() {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, APP_NAME.c_str(), nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	// TODO
	// recreate swapchain upon resizing the window
	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
	}

	// initialize vulkan
	void initVulkan() {
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
	}

	// create vulkan instance for all
	void createInstance() {
		VkApplicationInfo appInfo{
			.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pApplicationName = APP_NAME.c_str(),
			.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
			.pEngineName = "No Engine",
			.engineVersion = VK_MAKE_VERSION(1, 0, 0),
			.apiVersion = VK_API_VERSION_1_0,
		};

		if (enableValidationLayers) {
			UT_CHECK_ERR(checkRequiredLayersSupport(validationLayers), "do not support validation layers");
		}
		auto extensions = getRequiredExtensions();
		VkInstanceCreateInfo createInfo{
			.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pApplicationInfo = &appInfo,
			.enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
			.ppEnabledExtensionNames = extensions.data(),
		};
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = validationLayers.size();
			createInfo.ppEnabledLayerNames = validationLayers.data();
			VkDebugUtilsMessengerCreateInfoEXT createInfo;
			populateDebugMessengerCreateInfo(createInfo);
			createInfo.pNext = (VkDebugUtilsMessengerEXT*)&createInfo;
		} else {
			createInfo.enabledLayerCount = 0;
			createInfo.pNext = nullptr;
		}

		UT_CHECK_ERR(vkCreateInstance(&createInfo, nullptr, &instance) == VK_SUCCESS, "failed to create instace");
	}

	// setup the debug messenger, the debug extension must be enabled
	void setupDebugMessenger() {
		if (!enableValidationLayers) return;
		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);
		UT_CHECK_ERR(CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) == VK_SUCCESS, "failed to setup debug messenger, check whether debug ext is not supported");
	}

	// create surface and bind it to the instance
	void createSurface() {
		UT_CHECK_ERR(glfwCreateWindowSurface(instance, window, nullptr, &surface) == VK_SUCCESS, "failed to create window surface");
	}

	// choose a suitable device
	void pickPhysicalDevice() {
		uint32_t cnt = 0;
		vkEnumeratePhysicalDevices(instance, &cnt, nullptr);
		UT_CHECK_ERR(cnt != 0, "failed to find GPUs with Vulkan support");
		
		std::vector<VkPhysicalDevice> devices(cnt);
		vkEnumeratePhysicalDevices(instance, &cnt, devices.data());
		for (const auto& device : devices) {
			VkPhysicalDeviceProperties deviceProperties;
			vkGetPhysicalDeviceProperties(device, &deviceProperties);
			indices = findQueueFamilyProperties(device, surface);
			details = querySwapchainSupport(device, surface);
			if (isDeviceSuitable(device, surface)) {
				physicalDevice = device;
				// TODO: add msaa
				break;
			}
		}
		UT_CHECK_ERR(physicalDevice != VK_NULL_HANDLE, "failed to find a suitabale GPU");
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	void cleanup() {
		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();
	}
};


// get glfw & validation extensions
static std::vector<const char*> getRequiredExtensions() {
	uint32_t cnt = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&cnt);
	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + cnt);
	if (enableValidationLayers) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
	return extensions;
}

// chech if the device supports swapchain, extensions and features
static bool isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR& surface) {
	QueueFamilyIndices indices = findQueueFamilyProperties(device, surface);
	bool extensionsSupported = checkRequiredExtensionsSupport(device, deviceExtensions);
	bool swapChainAdequate = false;
	if (extensionsSupported) { // swapchain is included
		SwapchainSupportDetails swapchainSupport = querySwapchainSupport(device, surface);
		swapChainAdequate = !swapchainSupport.formats.empty() && !swapchainSupport.presentModes.empty();
	}
	VkPhysicalDeviceFeatures supportedFeatures;
	vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
	// TODO: check samplerAnisotropy if msaa is required
	return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

static SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice device, VkSurfaceKHR& surface) {
	SwapchainSupportDetails details;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
	uint32_t cnt;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &cnt, nullptr);
	if (cnt != 0) {
		details.formats.resize(cnt);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &cnt, details.formats.data());
	}
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &cnt, nullptr);
	if (cnt != 0) {
		details.presentModes.resize(cnt);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &cnt, details.presentModes.data());
	}
	return details;
}

static QueueFamilyIndices findQueueFamilyProperties(VkPhysicalDevice device, VkSurfaceKHR& surface) {
	QueueFamilyIndices indices;
	uint32_t cnt;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &cnt, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(cnt);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &cnt, queueFamilies.data());
	int i = 0;
	for (const auto& queueFamily : queueFamilies) {
		if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
			indices.uniformFamily = i;
		}
		VkBool32 presentSupport = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
		if (presentSupport) {
			indices.presentFamily = i;
		}
		++i;
	}
	return indices;
}


int main() {
	App app;
	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
